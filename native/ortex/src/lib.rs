//! # Ortex
//! Rust bindings between [ONNX Runtime](https://github.com/microsoft/onnxruntime) and
//! Erlang/Elixir using [Ort](https://docs.rs/ort) and [Rustler](https://docs.rs/rustler).
//! These are only meant to be accessed via the NIF interface provided by Rustler and not
//! directly.

mod constants;
mod cuda_preprocess;
mod image;
mod model;
mod tensor;
mod utils;

use cuda_preprocess::CudaPreprocessedImage;
use model::OrtexModel;
use rustler::Resource;
use tensor::OrtexTensor;

impl Resource for CudaPreprocessedImage {}

use rustler::types::Binary;
use rustler::ResourceArc;
use rustler::{Atom, Env, NifResult, Term};

#[rustler::nif(schedule = "DirtyIo")]
fn init(
    env: Env,
    model_path: String,
    eps: Vec<(Atom, Vec<(String, String)>)>,
    opt: i32,
) -> NifResult<ResourceArc<model::OrtexModel>> {
    let eps = utils::map_eps(env, eps);
    let model = model::init(model_path, eps, opt)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(ResourceArc::new(model))
}

#[rustler::nif(schedule = "DirtyIo")]
fn unload(model: ResourceArc<model::OrtexModel>) {
    model::unload(model);
}

#[rustler::nif]
fn show_session(
    model: ResourceArc<model::OrtexModel>,
) -> NifResult<(
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
)> {
    model::show(model).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyIo")]
fn run(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> NifResult<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>> {
    model::run(model, inputs).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyIo")]
fn run_binary<'a>(
    env: Env<'a>,
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<(Binary<'a>, Vec<usize>, String, usize)>,
) -> NifResult<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>> {
    let _ = env;
    model::run_binary(model, &inputs).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyIo")]
fn run_cuda(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<(u64, Vec<i64>, String, usize, i32)>,
) -> NifResult<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>> {
    model::run_cuda(model, &inputs).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn from_binary(bin: Binary, shape: Term, dtype: Term) -> NifResult<ResourceArc<OrtexTensor>> {
    let shape: Vec<usize> = rustler::types::tuple::get_tuple(shape)?
        .iter()
        .map(|x| -> NifResult<usize> { Ok(x.decode::<usize>())? })
        .collect::<NifResult<Vec<usize>>>()?;
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;

    utils::from_binary(bin, shape, dtype_str, dtype_bits)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    utils::to_binary(env, reference, bits, limit)
}

#[rustler::nif]
pub fn slice<'a>(
    tensor: ResourceArc<OrtexTensor>,
    start_indicies: Vec<isize>,
    lengths: Vec<isize>,
    strides: Vec<isize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.slice(
        start_indicies,
        lengths,
        strides,
    )))
}

#[rustler::nif]
pub fn reshape<'a>(
    tensor: ResourceArc<OrtexTensor>,
    shape: Vec<usize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.reshape(shape)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn prepare_image<'a>(
    env: Env<'a>,
    bin: Binary,
    width: u32,
    height: u32,
    size: u32,
) -> NifResult<Term<'a>> {
    image::prepare_image(env, bin, width, height, size)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn prepare_resized_image<'a>(
    env: Env<'a>,
    bin: Binary,
    scaled_width: u32,
    scaled_height: u32,
    canvas_size: u32,
    pad_x: u32,
    pad_y: u32,
    pad_value: u8,
) -> NifResult<Term<'a>> {
    image::prepare_resized_image(env, bin, scaled_width, scaled_height, canvas_size, pad_x, pad_y, pad_value)
}

///  normalize/transpose/pad on GPU via a custom CUDA kernel
#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
pub fn prepare_resized_image_cuda(
    bin: Binary,
    scaled_width: i32,
    scaled_height: i32,
    canvas_width: i32,
    canvas_height: i32,
    pad_x: i32,
    pad_y: i32,
    pad_value: u8,
    half: bool,
) -> NifResult<(u64, Vec<i64>, i32, usize, ResourceArc<CudaPreprocessedImage>)> {
    let (ptr, shape, device_ordinal, dtype_bits, keepalive) =
        cuda_preprocess::prepare_resized_bgr_cuda(
            bin.as_slice(),
            scaled_width,
            scaled_height,
            canvas_width,
            canvas_height,
            pad_x,
            pad_y,
            pad_value,
            half,
        )
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;

    Ok((
        ptr,
        shape,
        device_ordinal,
        dtype_bits,
        ResourceArc::new(keepalive),
    ))
}

// Sigmoid over the full prototype array is real CPU work, not sub-ms.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn create_mask<'a>(
    env: Env<'a>,
    coefficients: Vec<f32>,
    prototypes_bin: rustler::Binary,
    proto_shape_term: Term<'a>, // Elixir tuple {batch, m, h, w} -> Vec<usize>
    dtype_bits: usize,
    threshold: f32,
) -> Result<Term<'a>, rustler::Error> {
    model::create_mask(
        env,
        coefficients,
        prototypes_bin,
        proto_shape_term,
        dtype_bits,
        threshold,
    )
}

#[rustler::nif]
pub fn concatenate<'a>(
    tensors: Vec<ResourceArc<OrtexTensor>>,
    dtype: Term,
    axis: i32,
) -> NifResult<ResourceArc<OrtexTensor>> {
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;
    let concatted = tensor::concatenate(tensors, (&dtype_str, dtype_bits), axis as usize);
    Ok(ResourceArc::new(concatted))
}

pub fn on_load(env: Env) -> bool {
    tracing_subscriber::fmt::init();
    env.register::<OrtexModel>().is_ok()
        && env.register::<OrtexTensor>().is_ok()
        && env.register::<CudaPreprocessedImage>().is_ok()
}

rustler::init!(
    "Elixir.Ortex.Native",
    load = |env: Env, _term: Term| -> bool { on_load(env) }
);
