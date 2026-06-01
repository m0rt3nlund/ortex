//! # Ortex
//! Rust bindings between [ONNX Runtime](https://github.com/microsoft/onnxruntime) and
//! Erlang/Elixir using [Ort](https://docs.rs/ort) and [Rustler](https://docs.rs/rustler).
//! These are only meant to be accessed via the NIF interface provided by Rustler and not
//! directly.

mod constants;
mod image;
mod model;
mod tensor;
mod utils;

use model::OrtexModel;
use tensor::OrtexTensor;

use rustler::types::Binary;
use rustler::ResourceArc;
use rustler::{Atom, Env, NifResult, Term};

#[rustler::nif(schedule = "DirtyIo")]
fn init(
    env: Env,
    model_path: String,
    eps: Vec<Atom>,
    opt: i32,
) -> NifResult<ResourceArc<model::OrtexModel>> {
    let eps = utils::map_eps(env, eps);
    let model = model::init(model_path, eps, opt)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(ResourceArc::new(model))
}

#[rustler::nif]
fn show_session(
    model: ResourceArc<model::OrtexModel>,
) -> NifResult<(
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
)> {
    Ok(model::show(model))
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

#[rustler::nif]
pub fn prepare_image<'a>(
    env: Env<'a>,
    bin: Binary,
    width: u32,
    height: u32,
    size: u32,
) -> NifResult<Term<'a>> {
    image::prepare_image(env, bin, width, height, size)
}

#[rustler::nif]
pub fn create_mask<'a>(
    env: Env<'a>,
    coefficients: Vec<f32>,
    prototypes_bin: rustler::Binary,
    proto_shape_term: Term<'a>, // Elixir tuple {batch, m, h, w} -> Vec<usize>
    threshold: f32,
) -> Result<Term<'a>, rustler::Error> {
    model::create_mask(
        env,
        coefficients,
        prototypes_bin,
        proto_shape_term,
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
    env.register::<OrtexModel>().is_ok() && env.register::<OrtexTensor>().is_ok()
}

rustler::init!(
    "Elixir.Ortex.Native",
    load = |env: Env, _term: Term| -> bool { on_load(env) }
);
