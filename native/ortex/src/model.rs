//! Abstractions for creating an ONNX Runtime Session and Environment
//!  which can be safely passed to and from the BEAM.

use crate::tensor::OrtexTensor;
use crate::utils::{is_bool_input, map_opt_level};
use ndarray::{Array, ArrayView, IxDyn};
use std::convert::TryInto;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{Shape, TensorElementType};
use ort::value::{DynTensor, TensorRef, TensorRefMut, ValueType};
use ort::Error;
use rustler::types::Binary;
use rustler::Atom;
use rustler::Resource;
use rustler::ResourceArc;
use std::error::Error as StdError;
use std::sync::Mutex;

/// Holds the model's ONNX Runtime session
pub struct OrtexModel {
    pub session: Mutex<Session>,
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

fn tensor_element_dtype(ty: TensorElementType) -> Result<(String, usize), Box<dyn StdError + Send + Sync>> {
    match ty {
        TensorElementType::Float32 => Ok(("f".to_string(), 32)),
        TensorElementType::Float16 => Ok(("f".to_string(), 16)),
        other => Err(format!("unsupported cuda output dtype: {other:?}").into()),
    }
}

/// The execution providers are Atoms from Erlang/Elixir.
pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    opt: i32,
) -> Result<OrtexModel, Error> {
    let session = Session::builder()?
        .with_optimization_level(map_opt_level(opt))?
        .with_execution_providers(eps)?
        .commit_from_file(model_path)?;

    Ok(OrtexModel {
        session: Mutex::new(session),
    })
}

pub fn show(
    model: ResourceArc<OrtexModel>,
) -> (
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
) {
    let session = model.session.lock().unwrap();

    let mut inputs = Vec::new();
    for input in session.inputs() {
        let name = input.name().to_string();
        let repr = format!("{:#?}", input.dtype());
        let dims: Option<Vec<i64>> = input.dtype().tensor_shape().map(|s| s.to_vec());
        inputs.push((name, repr, dims));
    }

    let mut outputs = Vec::new();
    for output in session.outputs() {
        let name = output.name().to_string();
        let repr = format!("{:#?}", output.dtype());
        let dims: Option<Vec<i64>> = output.dtype().tensor_shape().map(|s| s.to_vec());
        outputs.push((name, repr, dims));
    }

    (inputs, outputs)
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError + Send + Sync>>
{
    let mut session = model.session.lock().unwrap();

    // Bool-converted temporaries must outlive ortified_inputs since inputs borrow from them.
    let bool_converted: Vec<OrtexTensor> = inputs
        .iter()
        .zip(session.inputs())
        .filter_map(|(elixir_input, onnx_input)| {
            if is_bool_input(onnx_input.dtype()) {
                Some((&**elixir_input).clone().to_bool())
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
    let mut collected_outputs = Vec::new();

    for output_name in outputs.keys() {
        let val = outputs.get(output_name).expect(
            &format!(
                "Expected {} to be in the outputs, but didn't find it",
                output_name
            )[..],
        );

        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();
        let collected_output = (ResourceArc::new(ortextensor), shape, dtype, bits);
        collected_outputs.push(collected_output);
    }

    Ok(collected_outputs)
}

pub fn run_binary<'a>(
    model: ResourceArc<OrtexModel>,
    inputs: &[(Binary<'a>, Vec<usize>, String, usize)],
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError + Send + Sync>>
{
    let mut session = model.session.lock().unwrap();
    let mut ortified_inputs: Vec<ort::session::SessionInputValue<'_>> = Vec::new();

    for ((bin, shape, dtype_str, dtype_bits), onnx_input) in inputs.iter().zip(session.inputs()) {
        let n: usize = shape.iter().product();
        let ptr = bin.as_ptr();

        if is_bool_input(onnx_input.dtype()) {
            let u8_slice = unsafe { std::slice::from_raw_parts(ptr, n) };
            let bool_vec: Vec<bool> = u8_slice.iter().map(|&x| x != 0).collect();
            let arr = Array::from_shape_vec(IxDyn(shape.as_slice()), bool_vec)?;
            ortified_inputs.push(ort::value::Value::from_array(arr)?.into());
            continue;
        }

        macro_rules! make_input {
            ($t:ty) => {{
                let slice: &[$t] = unsafe { std::slice::from_raw_parts(ptr as *const $t, n) };
                let arr = ArrayView::<$t, IxDyn>::from_shape(IxDyn(shape.as_slice()), slice)?;
                TensorRef::<$t>::from_array_view(arr)?.into()
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

    let mut collected_outputs = Vec::new();
    for output_name in outputs.keys() {
        let val = outputs.get(output_name).expect("output key missing");
        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();
        collected_outputs.push((ResourceArc::new(ortextensor), shape, dtype, bits));
    }

    Ok(collected_outputs)
}

pub fn run_cuda(
    model: ResourceArc<OrtexModel>,
    inputs: &[(u64, Vec<i64>, String, usize, i32)],
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError + Send + Sync>>
{
    let mut session = model.session.lock().unwrap();
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

    let mut collected_outputs = Vec::new();
    for output_name in outputs.keys() {
        let val = outputs.get(output_name).expect("output key missing");
        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();
        collected_outputs.push((ResourceArc::new(ortextensor), shape, dtype, bits));
    }

    Ok(collected_outputs)
}

/// Same CUDA device-pointer inputs as `run_cuda`, but additionally pins each
/// output named in `cuda_output_names` to CUDA device memory via IoBinding
/// instead of letting ONNX Runtime copy it to the host.
#[allow(clippy::too_many_arguments)]
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
    Box<dyn StdError + Send + Sync>,
> {
    let mut session = model.session.lock().unwrap();

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

            cuda_outputs.push((
                name,
                ptr,
                shape_vec,
                dtype_str,
                dtype_bits,
                cuda_device_index,
                ResourceArc::new(OrtexCudaOutput { value: dyn_tensor }),
            ));
        } else {
            let ortextensor: OrtexTensor = (&val).try_into()?;
            let shape = ortextensor.shape();
            let (dtype, bits) = ortextensor.dtype();
            host_outputs.push((name, ResourceArc::new(ortextensor), shape, dtype, bits));
        }
    }

    Ok((host_outputs, cuda_outputs))
}
