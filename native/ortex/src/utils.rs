//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;
use ndarray::{Array, IxDyn};

use rustler::types::Binary;
use rustler::ResourceArc;
use rustler::{Atom, Env, NifResult};

use ort::ep::cuda::ConvAlgorithmSearch;
use ort::ep::{ArenaExtendStrategy, ExecutionProviderDispatch};
use ort::session::builder::GraphOptimizationLevel;

/// Copies a binary into an owned array of `T`. Checks that the binary holds enough bytes for `shape`, and copies byte
/// by byte, so a binary that is not aligned for `T` (e.g. a sub-binary) is fine too.
/// `T` must be a plain number type (any bit pattern is a valid value).
pub fn copy_from_binary<T: Copy>(bytes: &[u8], shape: &[usize]) -> Result<Array<T, IxDyn>, String> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or("tensor shape is too large")?;
    let size = n
        .checked_mul(std::mem::size_of::<T>())
        .ok_or("tensor shape is too large")?;
    if bytes.len() < size {
        return Err(format!(
            "binary has {} bytes, but shape {:?} needs {}",
            bytes.len(),
            shape,
            size
        ));
    }
    let mut data: Vec<T> = Vec::with_capacity(n);
    // Safety: `data` has room for `n` values (`size` bytes), `bytes` has at least `size` bytes, and every bit pattern
    // is a valid `T`
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.as_mut_ptr() as *mut u8, size);
        data.set_len(n);
    }
    Array::from_shape_vec(IxDyn(shape), data).map_err(|e| e.to_string())
}

/// Given a Binary term, shape, and dtype from the BEAM, constructs an OrtexTensor
pub fn from_binary(
    bin: Binary,
    shape: Vec<usize>,
    dtype_str: String,
    dtype_bits: usize,
) -> Result<ResourceArc<OrtexTensor>, String> {
    let bytes = bin.as_slice();
    let tensor = match (dtype_str.as_ref(), dtype_bits) {
        ("bf", 16) => OrtexTensor::bf16(copy_from_binary(bytes, &shape)?),
        ("f", 16) => OrtexTensor::f16(copy_from_binary(bytes, &shape)?),
        ("f", 32) => OrtexTensor::f32(copy_from_binary(bytes, &shape)?),
        ("f", 64) => OrtexTensor::f64(copy_from_binary(bytes, &shape)?),
        ("s", 8) => OrtexTensor::s8(copy_from_binary(bytes, &shape)?),
        ("s", 16) => OrtexTensor::s16(copy_from_binary(bytes, &shape)?),
        ("s", 32) => OrtexTensor::s32(copy_from_binary(bytes, &shape)?),
        ("s", 64) => OrtexTensor::s64(copy_from_binary(bytes, &shape)?),
        ("u", 8) => OrtexTensor::u8(copy_from_binary(bytes, &shape)?),
        ("u", 16) => OrtexTensor::u16(copy_from_binary(bytes, &shape)?),
        ("u", 32) => OrtexTensor::u32(copy_from_binary(bytes, &shape)?),
        ("u", 64) => OrtexTensor::u64(copy_from_binary(bytes, &shape)?),
        _ => return Err(format!("unsupported dtype ({}, {})", dtype_str, dtype_bits)),
    };
    Ok(ResourceArc::new(tensor))
}

/// Given a reference to an OrtexTensor return the binary representation to be used
/// by the BinaryBackend of Nx.
pub fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    _bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    // Only the first `limit` elements (`inspect` asks for a few); the binary is a view of the tensor's data
    Ok(reference.make_binary(env, |x| {
        let bytes = x.to_bytes();
        let (_, bits) = x.dtype();
        &bytes[..limit.saturating_mul(bits / 8).min(bytes.len())]
    }))
}

/// Takes a vec of (Atom, options) pairs and transforms them into ExecutionProviderDispatches.
/// Options are flat string key-value pairs so they can be passed generically from Elixir.
pub fn map_eps(
    env: rustler::env::Env,
    eps: Vec<(Atom, Vec<(String, String)>)>,
) -> Vec<ExecutionProviderDispatch> {
    eps.into_iter()
        .map(
            |(e, opts)| match &e.to_term(env).atom_to_string().unwrap()[..] {
                CPU => ort::ep::cpu::CPU::default().build(),
                CUDA => {
                    let mut device_id: i32 = 0;
                    let mut memory_limit: Option<usize> = None;
                    let mut arena_extend_strategy: Option<ArenaExtendStrategy> = None;
                    let mut conv_algo_search: Option<ConvAlgorithmSearch> = None;
                    let mut conv_max_workspace: Option<bool> = None;
                    for (k, v) in &opts {
                        match k.as_str() {
                            "device_id" => device_id = v.parse().unwrap_or(device_id),
                            "gpu_mem_limit" => memory_limit = v.parse().ok(),
                            "arena_extend_strategy" => {
                                arena_extend_strategy = match v.as_str() {
                                    "same_as_requested" => Some(ArenaExtendStrategy::SameAsRequested),
                                    "next_power_of_two" => Some(ArenaExtendStrategy::NextPowerOfTwo),
                                    _ => None,
                                }
                            }
                            "cudnn_conv_algo_search" => {
                                conv_algo_search = match v.as_str() {
                                    "heuristic" => Some(ConvAlgorithmSearch::Heuristic),
                                    "exhaustive" => Some(ConvAlgorithmSearch::Exhaustive),
                                    "default" => Some(ConvAlgorithmSearch::Default),
                                    _ => None,
                                }
                            }
                            "cudnn_conv_use_max_workspace" => conv_max_workspace = Some(v == "true"),
                            _ => {}
                        }
                    }
                    let mut ep =
                        ort::ep::cuda::CUDA::default().with_device_id(device_id);
                    if let Some(limit) = memory_limit {
                        ep = ep.with_memory_limit(limit);
                    }
                    if let Some(strategy) = arena_extend_strategy {
                        ep = ep.with_arena_extend_strategy(strategy);
                    }
                    if let Some(search) = conv_algo_search {
                        ep = ep.with_conv_algorithm_search(search);
                    }
                    if let Some(enable) = conv_max_workspace {
                        ep = ep.with_conv_max_workspace(enable);
                    }
                    ep.build().error_on_failure()
                }
                TENSORRT => {
                    let mut workspace_size: usize = 1_073_741_824;
                    let mut engine_cache = false;
                    let mut engine_cache_path = String::new();
                    let mut fp16 = false;
                    let mut int8 = false;
                    for (k, v) in &opts {
                        match k.as_str() {
                            "max_workspace_size" => {
                                workspace_size = v.parse().unwrap_or(workspace_size)
                            }
                            "engine_cache" => engine_cache = v == "true",
                            "engine_cache_path" => engine_cache_path = v.clone(),
                            "fp16" => fp16 = v == "true",
                            "int8" => int8 = v == "true",
                            _ => {}
                        }
                    }
                    let ep =
                        ort::ep::tensorrt::TensorRT::default()
                            .with_max_workspace_size(workspace_size)
                            .with_engine_cache(engine_cache)
                            .with_fp16(fp16)
                            .with_int8(int8);
                    let ep = if engine_cache_path.is_empty() {
                        ep
                    } else {
                        ep.with_engine_cache_path(engine_cache_path)
                    };
                    ep.build().error_on_failure()
                }
                #[cfg(feature = "acl")]
                ACL => ort::ep::acl::ACL::default().build(),
                #[cfg(feature = "onednn")]
                ONEDNN => ort::ep::onednn::OneDNN::default().build(),
                #[cfg(feature = "coreml")]
                COREML => ort::ep::coreml::CoreML::default().build(),
                #[cfg(windows)]
                DIRECTML => {
                    ort::ep::directml::DirectML::default()
                        .build()
                        .error_on_failure()
                }
                #[cfg(feature = "rocm")]
                ROCM => ort::ep::rocm::ROCm::default().build(),
                _ => ort::ep::cpu::CPU::default().build(),
            },
        )
        .collect()
}

/// Take an optimization level and returns the
pub fn map_opt_level(opt: i32) -> GraphOptimizationLevel {
    match opt {
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        3 => GraphOptimizationLevel::Level3,
        _ => GraphOptimizationLevel::Disable,
    }
}

pub fn is_bool_input(inp: &ort::value::ValueType) -> bool {
    match inp {
        ort::value::ValueType::Tensor { ty, .. } => ty == &ort::value::TensorElementType::Bool,
        ort::value::ValueType::Map { value, .. } => value == &ort::value::TensorElementType::Bool,
        ort::value::ValueType::Sequence(boxed_input) => is_bool_input(boxed_input),
        ort::value::ValueType::Optional(boxed_input) => is_bool_input(boxed_input),
    }
}
