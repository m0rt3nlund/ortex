//! Abstractions for creating an ONNX Runtime Session and Environment
//!  which can be safely passed to and from the BEAM.

use crate::tensor::OrtexTensor;
use crate::utils::{copy_from_binary, is_bool_input, map_opt_level};
use ndarray::{ArrayView, IxDyn};
use std::convert::TryInto;

use ort::ep::ExecutionProviderDispatch;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{Shape, TensorElementType};
use ort::value::{DynTensor, TensorRef, TensorRefMut, ValueType};
use rustler::types::Binary;
use rustler::Atom;
use rustler::Resource;
use rustler::ResourceArc;
use std::error::Error as StdError;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::OnceLock;

type BoxError = Box<dyn StdError + Send + Sync>;

/// The names, types and dimensions of a session's inputs or outputs
type Signature = Vec<(String, String, Option<Vec<i64>>)>;

/// Work for a model's thread, run with its session
type Job = Box<dyn FnOnce(&mut Session) + Send + 'static>;

/// Holds the model's ONNX Runtime session, on a thread of its own: the session is made, run and dropped there only.
///
/// The CUDA execution provider sets up cuBLAS and cuDNN handles (with their workspaces on the GPU) for every thread
/// that runs a session. The run NIFs are dirty, so any of the BEAM's dirty schedulers could run a session, and every
/// session ended up with a set of handles per scheduler. On one thread it has one set. A session ran one inference at
/// a time before as well (behind a Mutex), different models still run at the same time.
pub struct OrtexModel {
    jobs: Mutex<mpsc::Sender<Job>>,
    inputs: Signature,
    outputs: Signature,
}
impl Resource for OrtexModel {}

/// Owns an ONNX Runtime output `Value` that was bound to CUDA device memory via
/// IoBinding (see `run_cuda_pinned`), keeping its underlying device buffer alive
/// for as long as Elixir holds a reference to this resource.
pub struct OrtexCudaOutput {
    #[allow(dead_code)]
    pub value: DynTensor,
}
// Safety: `value` only ever exposes a raw device pointer (via `data_ptr`) read
// by native code on the CUDA-bound thread that consumes it.
unsafe impl Send for OrtexCudaOutput {}
unsafe impl Sync for OrtexCudaOutput {}
impl Resource for OrtexCudaOutput {}
// `ort::Value` holds a `Box<dyn Any>` internally (its optional backing store),
// and auto traits don't propagate through an unconstrained trait object.
impl std::panic::RefUnwindSafe for OrtexCudaOutput {}

/// Moves what it holds to the model's thread. Safety: the thread that sends it waits until the model's thread is done
/// with it (see `OrtexModel::with_session`), so the two never use it at the same time
struct AssertSend<T>(T);
unsafe impl<T> Send for AssertSend<T> {}
impl<T> AssertSend<T> {
    // A method, so a closure captures the whole wrapper and not only the field
    fn into_inner(self) -> T {
        self.0
    }
}

/// More inputs than the model takes would be dropped without a word (they are matched to the model's in order)
fn check_input_count(session: &Session, given: usize) -> Result<(), BoxError> {
    let expected = session.inputs().len();
    if given > expected {
        return Err(format!("the model takes {expected} inputs, but {given} were given").into());
    }
    Ok(())
}

/// The outputs as `OrtexTensor`s, copied to the host, in the session's output order
fn collect_outputs(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<OrtexTensor>, BoxError> {
    let mut collected = Vec::with_capacity(outputs.len());
    for value in outputs.values() {
        collected.push((&*value).try_into()?);
    }
    Ok(collected)
}

fn tensor_element_dtype(ty: TensorElementType) -> Result<(String, usize), BoxError> {
    match ty {
        TensorElementType::Float32 => Ok(("f".to_string(), 32)),
        TensorElementType::Float16 => Ok(("f".to_string(), 16)),
        other => Err(format!("unsupported cuda output dtype: {other:?}").into()),
    }
}

impl OrtexModel {
    /// Runs `f` with the session on the model's thread, and waits for it
    fn with_session<'f, R: 'f>(&self, f: impl FnOnce(&mut Session) -> R + 'f) -> Result<R, BoxError> {
        let (done, result) = mpsc::sync_channel::<std::thread::Result<AssertSend<R>>>(1);
        let f = AssertSend(f);
        let job: Box<dyn FnOnce(&mut Session) + Send + 'f> = Box::new(move |session: &mut Session| {
            let f = f.into_inner();
            let _ = done.send(catch_unwind(AssertUnwindSafe(|| AssertSend(f(session)))));
        });
        // Safety: this waits below until the job has run, or was dropped without running, so what it borrows outlives
        // it. The job holds `done`, so `result` only gives up once the job is gone.
        let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce(&mut Session) + Send + 'f>, Job>(job) };

        let sent = self.jobs.lock().map_err(|_| "the model's queue is poisoned")?.send(job);
        if sent.is_err() {
            return Err("the model's thread has stopped".into());
        }
        match result.recv() {
            Ok(Ok(value)) => Ok(value.into_inner()),
            Ok(Err(panic)) => {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                Err(format!("the model panicked: {message}").into())
            }
            Err(_) => Err("the model's thread stopped while running".into()),
        }
    }
}

/// One CUDA memory pool for every session that asks for it (the CUDA provider's `shared_arena` option), instead of a
/// pool per session: memory one model gives back can be used by another. For trying a GPU with little memory, it is
/// off unless asked for. The pool is ONNX Runtime's environment's, so it is only given back when the application
/// stops, and the per-session `gpu_mem_limit` and `arena_extend_strategy` do not apply to it: it grows by what is
/// asked for (kSameAsRequested), up to `shared_arena_max_mem` bytes (0, the default, has no limit).
pub struct SharedArena {
    pub device_id: i32,
    pub max_mem: usize,
}

impl SharedArena {
    /// From the CUDA provider's options, when `shared_arena` is `true`
    pub fn from_options(options: &[(String, String)]) -> Option<SharedArena> {
        let value = |key: &str| options.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str());
        if value("shared_arena") != Some("true") {
            return None;
        }
        Some(SharedArena {
            device_id: value("device_id").and_then(|v| v.parse().ok()).unwrap_or(0),
            max_mem: value("shared_arena_max_mem").and_then(|v| v.parse().ok()).unwrap_or(0),
        })
    }
}

/// ONNX Runtime's environment, one per process (ort allows no more): every session is made with it
static ENVIRONMENT: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();

/// Makes the environment, when the NIF is loaded (`on_load`), or else the first time it is needed. Without telemetry:
/// ort turns it on by default, which in Microsoft's builds sends model and usage data
pub fn environment() -> Result<&'static ort::environment::Environment, BoxError> {
    match ENVIRONMENT.get_or_init(|| ort::init().with_telemetry(false).build().map_err(|e| e.to_string())) {
        Ok(environment) => Ok(environment),
        Err(message) => Err(format!("no ONNX Runtime environment: {message}").into()),
    }
}

/// The devices whose shared pool is registered, once each; the first session's `max_mem` is the one that counts
static SHARED_ARENAS: Mutex<Vec<i32>> = Mutex::new(Vec::new());

fn register_shared_arena(arena: &SharedArena) -> Result<(), BoxError> {
    use ort::AsPointer;

    let mut registered = SHARED_ARENAS.lock().map_err(|_| "the shared CUDA pool's lock is poisoned")?;
    if registered.contains(&arena.device_id) {
        return Ok(());
    }

    let environment = environment()?;
    let memory_info = MemoryInfo::new(AllocationDevice::CUDA, arena.device_id, AllocatorType::Arena, MemoryType::Default)?;
    let api = ort::api();

    // arena_extend_strategy 1 is kSameAsRequested
    let keys = [c"arena_extend_strategy".as_ptr(), c"max_mem".as_ptr()];
    let values: [usize; 2] = [1, arena.max_mem];
    let mut arena_cfg: *mut ort_sys::OrtArenaCfg = std::ptr::null_mut();
    check_status(unsafe { (api.CreateArenaCfgV2)(keys.as_ptr(), values.as_ptr(), keys.len(), &mut arena_cfg) })?;

    let device_id = std::ffi::CString::new(arena.device_id.to_string())?;
    let option_keys = [c"device_id".as_ptr()];
    let option_values = [device_id.as_ptr()];
    let status = unsafe {
        (api.CreateAndRegisterAllocatorV2)(
            environment.ptr().cast_mut(),
            c"CUDAExecutionProvider".as_ptr(),
            memory_info.ptr(),
            arena_cfg,
            option_keys.as_ptr(),
            option_values.as_ptr(),
            option_keys.len(),
        )
    };
    unsafe { (api.ReleaseArenaCfg)(arena_cfg) };
    check_status(status)?;

    registered.push(arena.device_id);
    Ok(())
}

fn check_status(status: ort_sys::OrtStatusPtr) -> Result<(), BoxError> {
    if status.0.is_null() {
        return Ok(());
    }
    let api = ort::api();
    let message = unsafe { std::ffi::CStr::from_ptr((api.GetErrorMessage)(status.0)) }
        .to_string_lossy()
        .into_owned();
    unsafe { (api.ReleaseStatus)(status.0) };
    Err(message.into())
}

/// Makes the session on a thread of its own, which then runs what `OrtexModel::with_session` gives it until the model
/// is dropped (its queue closes), and drops the session.
/// The execution providers are Atoms from Erlang/Elixir.
pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    opt: i32,
    shared_arena: Option<SharedArena>,
) -> Result<OrtexModel, BoxError> {
    let (jobs, queue) = mpsc::channel::<Job>();
    let (ready, started) = mpsc::sync_channel::<Result<(Signature, Signature), String>>(1);
    let build = AssertSend(move || -> Result<Session, BoxError> {
        // Since ort rc.15 the builder methods return `ort::Error<SessionBuilder>`, which carries the builder (for
        // recovery) and so is not Send/Sync; `ort::Error::from` drops it so `?` can box the error.
        let mut builder = Session::builder(environment()?)?
            .with_optimization_level(map_opt_level(opt))
            .map_err(ort::Error::<()>::from)?
            .with_execution_providers(eps)
            .map_err(ort::Error::<()>::from)?;
        if let Some(arena) = &shared_arena {
            register_shared_arena(arena)?;
            builder = builder.with_env_allocators().map_err(ort::Error::<()>::from)?;
        }
        Ok(builder.commit_from_file(model_path)?)
    });

    std::thread::Builder::new()
        .name("ortex-model".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let build = build.into_inner();
            let mut session = match build() {
                Ok(session) => session,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return;
                }
            };
            let mut inputs = Vec::new();
            for input in session.inputs() {
                let dims: Option<Vec<i64>> = input.dtype().tensor_shape().map(|s| s.to_vec());
                inputs.push((input.name().to_string(), format!("{:#?}", input.dtype()), dims));
            }
            let mut outputs = Vec::new();
            for output in session.outputs() {
                let dims: Option<Vec<i64>> = output.dtype().tensor_shape().map(|s| s.to_vec());
                outputs.push((output.name().to_string(), format!("{:#?}", output.dtype()), dims));
            }
            let _ = ready.send(Ok((inputs, outputs)));
            for job in queue {
                job(&mut session);
            }
        })?;

    match started.recv() {
        Ok(Ok((inputs, outputs))) => Ok(OrtexModel { jobs: Mutex::new(jobs), inputs, outputs }),
        Ok(Err(message)) => Err(message.into()),
        Err(_) => Err("the model's thread stopped while loading the model".into()),
    }
}

pub fn show(model: ResourceArc<OrtexModel>) -> (Signature, Signature) {
    (model.inputs.clone(), model.outputs.clone())
}

/// The outputs as the NIFs return them; made here, on the calling thread, not on the model's
fn pack(outputs: Vec<OrtexTensor>) -> Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)> {
    outputs
        .into_iter()
        .map(|tensor| {
            let shape = tensor.shape();
            let (dtype, bits) = tensor.dtype();
            (ResourceArc::new(tensor), shape, dtype, bits)
        })
        .collect()
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, BoxError> {
    let outputs = model.with_session(|session| -> Result<Vec<OrtexTensor>, BoxError> {
        check_input_count(session, inputs.len())?;
        // Bool-converted temporaries must outlive ortified_inputs since inputs borrow from them.
        let bool_converted: Vec<OrtexTensor> = inputs
            .iter()
            .zip(session.inputs())
            .filter_map(|(elixir_input, onnx_input)| {
                if is_bool_input(onnx_input.dtype()) {
                    Some(elixir_input.to_bool())
                } else {
                    None
                }
            })
            .collect();

        let mut bool_iter = bool_converted.iter();
        let mut ortified_inputs: Vec<ort::session::SessionInputValue<'_>> = Vec::new();

        for (elixir_input, onnx_input) in inputs.iter().zip(session.inputs()) {
            if is_bool_input(onnx_input.dtype()) {
                let v: ort::session::SessionInputValue<'_> = bool_iter.next().unwrap().try_into()?;
                ortified_inputs.push(v);
            } else {
                let v: ort::session::SessionInputValue<'_> = (&**elixir_input).try_into()?;
                ortified_inputs.push(v);
            }
        }

        let outputs = session.run(&ortified_inputs[..])?;
        collect_outputs(&outputs)
    })??;

    Ok(pack(outputs))
}

pub fn run_binary<'a>(
    model: ResourceArc<OrtexModel>,
    inputs: &[(Binary<'a>, Vec<usize>, String, usize)],
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, BoxError> {
    let outputs = model.with_session(|session| -> Result<Vec<OrtexTensor>, BoxError> {
        check_input_count(session, inputs.len())?;
        let mut ortified_inputs: Vec<ort::session::SessionInputValue<'_>> = Vec::new();

        for ((bin, shape, dtype_str, dtype_bits), onnx_input) in inputs.iter().zip(session.inputs()) {
            let bytes = bin.as_slice();

            if is_bool_input(onnx_input.dtype()) {
                let arr = copy_from_binary::<u8>(bytes, shape)?.mapv(|x| x != 0);
                ortified_inputs.push(ort::value::Value::from_array(arr)?.into());
                continue;
            }

            // Without a copy when the binary is aligned for the type, else (e.g. a sub-binary) copied.
            // copy_from_binary checks the binary is long enough for the shape.
            macro_rules! make_input {
                ($t:ty) => {{
                    if bytes.as_ptr() as usize % std::mem::align_of::<$t>() == 0 {
                        let n: usize = shape.iter().product();
                        if bytes.len() < n * std::mem::size_of::<$t>() {
                            return Err(format!(
                                "binary has {} bytes, but shape {:?} needs {}",
                                bytes.len(),
                                shape,
                                n * std::mem::size_of::<$t>()
                            )
                            .into());
                        }
                        // Safety: aligned, and long enough (checked above)
                        let slice: &[$t] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const $t, n) };
                        let arr = ArrayView::<$t, IxDyn>::from_shape(IxDyn(shape.as_slice()), slice)?;
                        TensorRef::<$t>::from_array_view(arr)?.into()
                    } else {
                        let arr = copy_from_binary::<$t>(bytes, shape)?;
                        ort::value::Value::from_array(arr)?.into()
                    }
                }};
            }

            let v: ort::session::SessionInputValue<'_> = match (dtype_str.as_ref(), *dtype_bits) {
                ("f", 32) => make_input!(f32),
                ("f", 64) => make_input!(f64),
                ("f", 16) => make_input!(half::f16),
                ("bf", 16) => make_input!(half::bf16),
                ("s", 8) => make_input!(i8),
                ("s", 16) => make_input!(i16),
                ("s", 32) => make_input!(i32),
                ("s", 64) => make_input!(i64),
                ("u", 8) => make_input!(u8),
                ("u", 16) => make_input!(u16),
                ("u", 32) => make_input!(u32),
                ("u", 64) => make_input!(u64),
                _ => return Err(format!("unsupported dtype ({}, {})", dtype_str, dtype_bits).into()),
            };
            ortified_inputs.push(v);
        }

        let outputs = session.run(&ortified_inputs[..])?;
        collect_outputs(&outputs)
    })??;

    Ok(pack(outputs))
}

pub fn run_cuda(
    model: ResourceArc<OrtexModel>,
    inputs: &[(u64, Vec<i64>, String, usize, i32)],
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, BoxError> {
    let outputs = model.with_session(|session| -> Result<Vec<OrtexTensor>, BoxError> {
        check_input_count(session, inputs.len())?;
        let mut ortified_inputs: Vec<ort::session::SessionInputValue<'_>> = Vec::new();

        for (ptr, shape, dtype_str, dtype_bits, device_index) in inputs.iter() {
            let info = MemoryInfo::new(AllocationDevice::CUDA, *device_index, AllocatorType::Device, MemoryType::Default)?;
            let data = *ptr as usize as *mut ort_sys::c_void;

            macro_rules! make_cuda_input {
                ($t:ty) => {{
                    let tensor_ref: TensorRefMut<'_, $t> =
                        unsafe { TensorRefMut::from_raw(info.clone(), data, Shape::from(shape.clone()))? };
                    tensor_ref.into()
                }};
            }

            let v: ort::session::SessionInputValue<'_> = match (dtype_str.as_ref(), *dtype_bits) {
                ("f", 32) => make_cuda_input!(f32),
                ("f", 16) => make_cuda_input!(half::f16),
                _ => return Err(format!("unsupported cuda dtype ({}, {})", dtype_str, dtype_bits).into()),
            };
            ortified_inputs.push(v);
        }

        let outputs = session.run(&ortified_inputs[..])?;
        collect_outputs(&outputs)
    })??;

    Ok(pack(outputs))
}

/// Same CUDA device-pointer inputs as `run_cuda`, but additionally pins each
/// output named in `cuda_output_names` to CUDA device memory via IoBinding
/// instead of letting ONNX Runtime copy it to the host.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn run_cuda_pinned(
    model: ResourceArc<OrtexModel>,
    inputs: &[(u64, Vec<i64>, String, usize, i32)],
    cuda_output_names: &[String],
    cuda_device_index: i32,
) -> Result<
    (
        Vec<(String, ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>,
        Vec<(String, u64, Vec<i64>, String, usize, i32, ResourceArc<OrtexCudaOutput>)>,
    ),
    BoxError,
> {
    let (host, cuda) = model.with_session(
        |session| -> Result<
            (
                Vec<(String, OrtexTensor)>,
                Vec<(String, u64, Vec<i64>, String, usize, OrtexCudaOutput)>,
            ),
            BoxError,
        > {
            check_input_count(session, inputs.len())?;
            let mut binding = session.create_binding()?;

            for ((ptr, shape, dtype_str, dtype_bits, device_index), onnx_input) in
                inputs.iter().zip(session.inputs())
            {
                let info = MemoryInfo::new(AllocationDevice::CUDA, *device_index, AllocatorType::Device, MemoryType::Default)?;
                let data = *ptr as usize as *mut ort_sys::c_void;

                macro_rules! bind_cuda_input {
                    ($t:ty) => {{
                        let tensor_ref: TensorRefMut<'_, $t> =
                            unsafe { TensorRefMut::from_raw(info.clone(), data, Shape::from(shape.clone()))? };
                        binding.bind_input(onnx_input.name(), &tensor_ref)?;
                    }};
                }

                match (dtype_str.as_ref(), *dtype_bits) {
                    ("f", 32) => bind_cuda_input!(f32),
                    ("f", 16) => bind_cuda_input!(half::f16),
                    _ => return Err(format!("unsupported cuda dtype ({}, {})", dtype_str, dtype_bits).into()),
                };
            }

            let cpu_info = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?;
            let cuda_out_info = MemoryInfo::new(AllocationDevice::CUDA, cuda_device_index, AllocatorType::Device, MemoryType::Default)?;

            for output in session.outputs() {
                if cuda_output_names.iter().any(|n| n == output.name()) {
                    binding.bind_output_to_device(output.name(), &cuda_out_info)?;
                } else {
                    binding.bind_output_to_device(output.name(), &cpu_info)?;
                }
            }

            let outputs = session.run_binding(&binding)?;

            let mut host_outputs = Vec::new();
            let mut cuda_outputs = Vec::new();

            for (name, val) in outputs {
                let name = name.to_string();

                if cuda_output_names.iter().any(|n| n == &name) {
                    let dyn_tensor: DynTensor = val.downcast()?;
                    let ptr = dyn_tensor.data_ptr() as u64;
                    let ValueType::Tensor { ty, shape, .. } = dyn_tensor.dtype() else {
                        return Err(format!("output `{name}` bound to CUDA memory is not a tensor").into());
                    };
                    let (dtype_str, dtype_bits) = tensor_element_dtype(*ty)?;
                    let shape_vec: Vec<i64> = shape.iter().copied().collect();

                    cuda_outputs.push((name, ptr, shape_vec, dtype_str, dtype_bits, OrtexCudaOutput { value: dyn_tensor }));
                } else {
                    let ortextensor: OrtexTensor = (&val).try_into()?;
                    host_outputs.push((name, ortextensor));
                }
            }

            Ok((host_outputs, cuda_outputs))
        },
    )??;

    let host_outputs = host
        .into_iter()
        .map(|(name, tensor)| {
            let shape = tensor.shape();
            let (dtype, bits) = tensor.dtype();
            (name, ResourceArc::new(tensor), shape, dtype, bits)
        })
        .collect();
    let cuda_outputs = cuda
        .into_iter()
        .map(|(name, ptr, shape, dtype_str, dtype_bits, output)| {
            (name, ptr, shape, dtype_str, dtype_bits, cuda_device_index, ResourceArc::new(output))
        })
        .collect();

    Ok((host_outputs, cuda_outputs))
}
