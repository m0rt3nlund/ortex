//! Abstractions for creating an ONNX Runtime Session and Environment which can be safely
//! passed to and from the BEAM.
//!
//! # Examples
//!
//! ```
//! let model = init("./models/resnet50.onnx", vec![])?;
//! let (inputs, outputs) = show(model)?;
//! ```

use crate::tensor::OrtexTensor;
use crate::utils::{is_bool_input, map_opt_level};
use ndarray::{s, Array, Array2, ArrayView, ArrayView3, IxDyn};
use std::convert::TryInto;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::tensor::Shape;
use ort::value::{TensorRef, TensorRefMut};
use ort::Error;
use rustler::types::Binary;
use rustler::Atom;
use rustler::Env;
use rustler::NewBinary;
use rustler::Resource;
use rustler::ResourceArc;
use rustler::Term;
use std::error::Error as StdError;
use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::thread;

type Job = Box<dyn FnOnce(&mut Session) + Send>;

/// Holds the model state which include onnxruntime session and environment. The session
/// itself, and everything that touches it (including creation), runs on one dedicated OS
/// thread so onnxruntime/CUDA never observes calls from more than one thread.
pub struct OrtexModel {
    tx: Mutex<Option<Sender<Job>>>,
    handle: Option<thread::JoinHandle<()>>,
}
impl Resource for OrtexModel {}

impl OrtexModel {
    fn with_session<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Session) -> R + Send + 'static,
        R: Send + 'static,
    {
        let (rtx, rrx) = channel();
        self.tx
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .send(Box::new(move |session| {
                let _ = rtx.send(f(session));
            }))
            .unwrap();
        rrx.recv().unwrap()
    }
}

impl Drop for OrtexModel {
    fn drop(&mut self) {
        self.tx.lock().unwrap().take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The execution providers are Atoms from Erlang/Elixir.
pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    opt: i32,
) -> Result<OrtexModel, Error> {
    let (tx, rx) = channel::<Job>();
    let (itx, irx) = channel();

    let handle = thread::spawn(move || {
        let built = Session::builder()
            .and_then(|b| b.with_optimization_level(map_opt_level(opt)))
            .and_then(|b| b.with_execution_providers(eps))
            .and_then(|b| b.commit_from_file(model_path));

        match built {
            Ok(mut session) => {
                itx.send(None).unwrap();
                for job in rx {
                    job(&mut session);
                }
            }
            Err(e) => itx.send(Some(e)).unwrap(),
        }
    });

    match irx.recv().unwrap() {
        Some(e) => Err(e),
        None => Ok(OrtexModel {
            tx: Mutex::new(Some(tx)),
            handle: Some(handle),
        }),
    }
}

pub fn show(
    model: ResourceArc<OrtexModel>,
) -> (
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
) {
    model.with_session(|session| {
        let mut inputs = Vec::new();
        for input in session.inputs.iter() {
            let name = input.name.to_string();
            let repr = format!("{:#?}", input.input_type);
            let dims: Option<Vec<i64>> = input.input_type.tensor_shape().map(|s| s.to_vec());
            inputs.push((name, repr, dims));
        }

        let mut outputs = Vec::new();
        for output in session.outputs.iter() {
            let name = output.name.to_string();
            let repr = format!("{:#?}", output.output_type);
            let dims: Option<Vec<i64>> = output.output_type.tensor_shape().map(|s| s.to_vec());
            outputs.push((name, repr, dims));
        }

        (inputs, outputs)
    })
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError + Send + Sync>>
{
    model.with_session(move |session| {
        // Bool-converted temporaries must outlive ortified_inputs since inputs borrow from them.
        let bool_converted: Vec<OrtexTensor> = inputs
            .iter()
            .zip(&session.inputs)
            .filter_map(|(elixir_input, onnx_input)| {
                if is_bool_input(&onnx_input.input_type) {
                    Some((&**elixir_input).clone().to_bool())
                } else {
                    None
                }
            })
            .collect();

        let mut bool_iter = bool_converted.iter();
        let mut ortified_inputs: Vec<ort::session::SessionInputValue<'_>> = Vec::new();

        for (elixir_input, onnx_input) in inputs.iter().zip(&session.inputs) {
            if is_bool_input(&onnx_input.input_type) {
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
    })
}

pub fn run_binary<'a>(
    model: ResourceArc<OrtexModel>,
    inputs: &[(Binary<'a>, Vec<usize>, String, usize)],
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError + Send + Sync>>
{
    let raw: Vec<(usize, Vec<usize>, String, usize)> = inputs
        .iter()
        .map(|(bin, shape, dtype_str, dtype_bits)| {
            (bin.as_ptr() as usize, shape.clone(), dtype_str.clone(), *dtype_bits)
        })
        .collect();

    model.with_session(move |session| {
        let mut ortified_inputs: Vec<ort::session::SessionInputValue<'_>> = Vec::new();

        for ((ptr, shape, dtype_str, dtype_bits), onnx_input) in raw.iter().zip(&session.inputs) {
            let n: usize = shape.iter().product();
            let ptr = *ptr as *const u8;

            if is_bool_input(&onnx_input.input_type) {
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
    })
}

pub fn run_cuda(
    model: ResourceArc<OrtexModel>,
    inputs: &[(u64, Vec<i64>, String, usize, i32)],
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError + Send + Sync>>
{
    let inputs = inputs.to_vec();

    model.with_session(move |session| {
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
    })
}

fn bytes_to_f32_vec(bytes: &[u8], dtype_bits: usize, count: usize) -> Result<Vec<f32>, String> {
    match dtype_bits {
        32 => Ok(
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count) }.to_vec(),
        ),
        16 => Ok(
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const half::f16, count) }
                .iter()
                .map(|x| x.to_f32())
                .collect(),
        ),
        other => Err(format!("unsupported dtype_bits: {other}")),
    }
}

pub fn create_mask<'a>(
    env: Env<'a>,
    // Decoded from Erlang terms (not reinterpreted raw bytes)
    coefficients: Vec<f32>,
    prototypes_bin: rustler::Binary,
    proto_shape_term: Term<'a>, // Elixir tuple {batch, m, h, w} -> Vec<usize>
    dtype_bits: usize,
    threshold: f32,
) -> Result<Term<'a>, rustler::Error> {
    // Decode as 4-element tuple directly
    let (batch_size, m, height, width): (usize, usize, usize, usize) =
        proto_shape_term.decode().map_err(|e| {
            println!("Failed to decode tuple: {:?}", e);
            rustler::Error::BadArg
        })?;

    // Validate 4D
    let expected_proto_size = batch_size * m * height * width;
    let elem_size = match dtype_bits {
        32 => std::mem::size_of::<f32>(),
        16 => std::mem::size_of::<half::f16>(),
        _ => {
            println!("Unsupported prototypes dtype_bits: {}", dtype_bits);
            return Err(rustler::Error::BadArg);
        }
    };

    // Validate coefficients
    if coefficients.len() != m {
        println!(
            "Invalid coefficients length: {}, expected: {}",
            coefficients.len(),
            m
        );
        return Err(rustler::Error::BadArg);
    }

    // Validate prototypes binary
    let protos_bytes = prototypes_bin.as_slice();
    if protos_bytes.len() != expected_proto_size * elem_size {
        println!(
            "Invalid prototypes binary size: {}, expected: {}",
            protos_bytes.len(),
            expected_proto_size * elem_size
        );
        return Err(rustler::Error::BadArg);
    }
    let protos_vec = bytes_to_f32_vec(protos_bytes, dtype_bits, expected_proto_size)
        .map_err(|_| rustler::Error::BadArg)?;
    let protos_slice: &[f32] = &protos_vec;

    // Assume single batch
    if batch_size != 1 {
        println!("Batch size not 1: {}", batch_size);
        return Err(rustler::Error::BadArg);
    }

    // Reshape to 3D view
    let proto_3d =
        ArrayView3::<f32>::from_shape((m, height, width), protos_slice).map_err(|e| {
            println!("ArrayView3 error: {:?}", e);
            rustler::Error::BadArg
        })?;

    // Weighted sum
    let mut output = Array2::<f32>::zeros((height, width));
    for i in 0..m {
        let proto_slice = proto_3d.slice(s![i, .., ..]);
        let weighted = proto_slice.mapv(|x| x * coefficients[i]);
        output += &weighted;
    }

    // Sigmoid and threshold
    let sigmoid = output.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let binary = sigmoid.mapv(|x| if x >= threshold { 255u8 } else { 0u8 });
    let binary_vec = binary
        .to_shape(height * width)
        .map_err(|e| {
            println!("Flatten error: {:?}", e);
            rustler::Error::BadArg
        })?
        .to_vec();

    // Return binary
    let mut new_bin = NewBinary::new(env, binary_vec.len());
    new_bin.as_mut_slice().copy_from_slice(&binary_vec);
    Ok(new_bin.into())
}

#[cfg(test)]
mod tests {
    use super::bytes_to_f32_vec;

    #[test]
    fn bytes_to_f32_vec_f32_roundtrip() {
        let values: [f32; 4] = [0.0, 1.5, -2.25, 3.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = bytes_to_f32_vec(&bytes, 32, values.len()).expect("f32 path should succeed");
        assert_eq!(out, values);
    }

    #[test]
    fn bytes_to_f32_vec_f16_widens_to_f32() {
        let values: [half::f16; 4] = [
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.5),
            half::f16::from_f32(-2.25),
            half::f16::from_f32(3.0),
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let out = bytes_to_f32_vec(&bytes, 16, values.len()).expect("f16 path should succeed");
        assert_eq!(out, vec![0.0f32, 1.5, -2.25, 3.0]);
    }

    #[test]
    fn bytes_to_f32_vec_rejects_unknown_dtype() {
        let bytes = [0u8; 8];
        assert!(bytes_to_f32_vec(&bytes, 8, 2).is_err());
    }
}
