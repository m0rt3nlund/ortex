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

use ort::execution_providers;
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::TensorRef;
use ort::Error;
use rustler::types::Binary;
use rustler::Atom;
use rustler::Env;
use rustler::NewBinary;
use rustler::Resource;
use rustler::ResourceArc;
use rustler::Term;
use std::error::Error as StdError;
use std::sync::Mutex;

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    pub session: Mutex<ort::session::Session>,
}
impl Resource for OrtexModel {}

// Since we're only using the session for inference and
// inference is threadsafe, this Sync is safe. Additionally,
// Environment is global and also threadsafe
// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Sync for OrtexModel {}

/// Creates a model given the path to the model and vector of execution providers.
/// The execution providers are Atoms from Erlang/Elixir.
pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    opt: i32,
) -> Result<OrtexModel, Error> {
    // TODO: send tracing logs to erlang/elixir _somehow_
    //tracing_subscriber::fmt::init();

    let session = Session::builder()?
        .with_optimization_level(map_opt_level(opt))?
        .with_execution_providers([execution_providers::TensorRTExecutionProvider::default()
            .with_max_workspace_size(2147483648)
            .with_engine_cache(true)
            .with_engine_cache_path("/opt/maskon/mvision")
            .with_fp16(true)
            .with_int8(false)
            .build()])?
        .commit_from_file(model_path)?;

    let state = OrtexModel {
        session: session.into(),
    };
    Ok(state)
}

/// Returns input/output information about a model. The result is a Tuple of
/// `inputs` and `outputs` with elements of `(Name, Type, Dimension)` where
/// `Dimension` elements of -1 are dynamic.
pub fn show(
    model: ResourceArc<OrtexModel>,
) -> (
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
) {
    let session: &mut ort::session::Session = &mut model.session.lock().unwrap();

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
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError>> {
    let session: &mut ort::session::Session = &mut model.session.lock().unwrap();

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
}

/// Like `run`, but takes raw BEAM binaries instead of OrtexTensor ResourceArcs.
/// Creates zero-copy TensorRef views directly from BEAM binary pointers, avoiding
/// the separate from_binary NIF call and the ndarray heap copy entirely.
pub fn run_binary<'a>(
    model: ResourceArc<OrtexModel>,
    inputs: &[(Binary<'a>, Vec<usize>, String, usize)],
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError>> {
    let session: &mut Session = &mut model.session.lock().unwrap();

    let mut ortified_inputs: Vec<ort::session::SessionInputValue<'a>> = Vec::new();

    for ((bin, shape, dtype_str, dtype_bits), onnx_input) in inputs.iter().zip(&session.inputs) {
        let n: usize = shape.iter().product();
        let ptr = bin.as_ptr();

        if is_bool_input(&onnx_input.input_type) {
            // Bool inputs require a type conversion; owned copy is unavoidable here.
            let u8_slice = &bin.as_slice()[..n];
            let bool_vec: Vec<bool> = u8_slice.iter().map(|&x| x != 0).collect();
            let arr = Array::from_shape_vec(IxDyn(shape.as_slice()), bool_vec)?;
            ortified_inputs.push(ort::value::Value::from_array(arr)?.into());
            continue;
        }

        macro_rules! make_input {
            ($t:ty) => {{
                // SAFETY: Binary<'a> is pinned by the BEAM for the duration of this NIF call.
                // The slice, ArrayView, and TensorRef all borrow with lifetime 'a which is
                // the NIF env lifetime, so they cannot outlive the binary data.
                let slice: &'a [$t] = unsafe {
                    std::slice::from_raw_parts(ptr as *const $t, n)
                };
                let arr = ArrayView::<$t, IxDyn>::from_shape(IxDyn(shape.as_slice()), slice)?;
                TensorRef::<$t>::from_array_view(arr)?.into()
            }};
        }

        let v: ort::session::SessionInputValue<'a> = match (dtype_str.as_ref(), *dtype_bits) {
            ("f", 32)  => make_input!(f32),
            ("f", 64)  => make_input!(f64),
            ("f", 16)  => make_input!(half::f16),
            ("bf", 16) => make_input!(half::bf16),
            ("s", 8)   => make_input!(i8),
            ("s", 16)  => make_input!(i16),
            ("s", 32)  => make_input!(i32),
            ("s", 64)  => make_input!(i64),
            ("u", 8)   => make_input!(u8),
            ("u", 16)  => make_input!(u16),
            ("u", 32)  => make_input!(u32),
            ("u", 64)  => make_input!(u64),
            _ => return Err(format!("unsupported dtype ({}, {})", dtype_str, dtype_bits).into()),
        };
        ortified_inputs.push(v);
    }

    let t_run = std::time::Instant::now();
    let outputs = session.run(&ortified_inputs[..])?;
    let t_extract = std::time::Instant::now();

    let mut collected_outputs = Vec::new();
    for output_name in outputs.keys() {
        let val = outputs.get(output_name).expect("output key missing");
        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();
        collected_outputs.push((ResourceArc::new(ortextensor), shape, dtype, bits));
    }

    let t_end = std::time::Instant::now();
    eprintln!(
        "[ortex run_binary] session.run={:?}  output_extract={:?}",
        t_extract - t_run,
        t_end - t_extract,
    );

    Ok(collected_outputs)
}

pub fn create_mask<'a>(
    env: Env<'a>,
    coefficients: Vec<f32>,
    prototypes_bin: rustler::Binary,
    proto_shape_term: Term<'a>, // Elixir tuple {batch, m, h, w} -> Vec<usize>
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
    let f32_size = std::mem::size_of::<f32>();

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
    if protos_bytes.len() != expected_proto_size * f32_size {
        println!(
            "Invalid prototypes binary size: {}, expected: {}",
            protos_bytes.len(),
            expected_proto_size * f32_size
        );
        return Err(rustler::Error::BadArg);
    }
    let protos_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(protos_bytes.as_ptr() as *const f32, expected_proto_size)
    };

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
