//! Custom CUDA kernel for the "rest of fit_rust" preprocessing step: BGR->RGB
//! swap, u8->f32 normalize, HWC->CHW transpose, and letterbox padding, fused
//! into one GPU kernel. Fed by the same CPU-side Evision resize the Torchx
//! path already uses (resize itself stays on CPU deliberately -- see
//! frame_scalers.ex -- since that avoids uploading the full camera frame).
//!
//! The kernel is compiled once via NVRTC and cached in a process-global
//! OnceLock; CudaContext::new(0) retains the device's *primary* context
//! (cuDevicePrimaryCtxRetain), the same context onnxruntime's CUDA EP and
//! Torchx already use in this process -- not a separate/duplicate one.
//!
//! The output buffer is likewise a single, reused, process-lifetime
//! allocation (behind a Mutex, resized only if the requested size changes)
//! rather than a fresh `alloc_zeros` per frame. A fresh-per-frame buffer was
//! the original design, wrapped in a Rustler ResourceArc for the BEAM to
//! free once unreachable -- but BEAM GC runs on its own schedule, not
//! deterministically, so unfreed GPU buffers could pile up between GC
//! passes and produce exactly the kind of occasional, seemingly-random
//! latency spike observed in practice (a GC pass finally freeing a batch of
//! them, stalling whatever CUDA call happened to be in flight at that
//! moment). Reusing one buffer removes that dependency entirely.
//!
//! Safety note: reusing a single buffer (rather than a small pool) is only
//! correct because the pipeline that calls this is currently strictly
//! sequential -- frame N+1's preprocessing never starts until frame N's
//! entire Inferencer.process/3 (preprocess -> Ortex.run -> postprocess) has
//! returned, so by the time this function's Mutex is next locked, the
//! previous frame's TensorRT read of this same buffer has already finished.
//! If the pipeline is ever changed to overlap frames (e.g. splitting
//! preprocess/infer into separate async GenStage stages), this needs to
//! become an actual pool (2-3 buffers, round-robin/refcounted) instead --
//! a single shared buffer would then be a genuine data race.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::{Arc, Mutex, OnceLock};

const KERNEL_SRC: &str = r#"
extern "C" __global__ void bgr_to_padded_rgb_chw(
    const unsigned char* src,
    float* dst,
    int scaled_width,
    int scaled_height,
    int canvas_size,
    int pad_x,
    int pad_y,
    float pad_value_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = canvas_size * canvas_size;
    if (idx >= total) return;

    int cy = idx / canvas_size;
    int cx = idx % canvas_size;
    int hw = canvas_size * canvas_size;

    int sx = cx - pad_x;
    int sy = cy - pad_y;

    if (sx >= 0 && sx < scaled_width && sy >= 0 && sy < scaled_height) {
        int src_offset = (sy * scaled_width + sx) * 3;
        unsigned char b = src[src_offset];
        unsigned char g = src[src_offset + 1];
        unsigned char r = src[src_offset + 2];

        dst[0 * hw + idx] = ((float)r) / 255.0f;
        dst[1 * hw + idx] = ((float)g) / 255.0f;
        dst[2 * hw + idx] = ((float)b) / 255.0f;
    } else {
        dst[0 * hw + idx] = pad_value_norm;
        dst[1 * hw + idx] = pad_value_norm;
        dst[2 * hw + idx] = pad_value_norm;
    }
}
"#;

struct KernelState {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
    // (len, buffer) -- reallocated only when a call requests a different len
    // than what's currently held (e.g. a different model's canvas_size).
    output: Mutex<Option<(usize, CudaSlice<f32>)>>,
}

// OnceLock rather than re-compiling/re-loading per call: NVRTC compilation
// and module loading are one-time, comparatively expensive setup costs that
// must not run on every frame.
static KERNEL: OnceLock<Result<KernelState, String>> = OnceLock::new();

fn kernel_state() -> Result<&'static KernelState, String> {
    KERNEL
        .get_or_init(|| {
            let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext::new failed: {e:?}"))?;
            let stream = ctx.default_stream();
            let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("NVRTC compile failed: {e:?}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("load_module failed: {e:?}"))?;
            let func = module
                .load_function("bgr_to_padded_rgb_chw")
                .map_err(|e| format!("load_function failed: {e:?}"))?;
            Ok(KernelState {
                ctx,
                stream,
                func,
                output: Mutex::new(None),
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Trivial marker kept only so the Elixir-facing NIF signature (and
/// Ortex.CudaTensor's `keepalive` field) doesn't need to change. The actual
/// GPU buffer is now owned by the static KernelState, not by this resource
/// -- its lifetime is the whole process, not tied to BEAM GC at all.
pub struct CudaPreprocessedImage;

/// Returns (raw_device_ptr, shape [1,3,canvas,canvas], device_ordinal, keepalive).
pub fn prepare_resized_bgr_cuda(
    bgr_u8: &[u8],
    scaled_width: i32,
    scaled_height: i32,
    canvas_size: i32,
    pad_x: i32,
    pad_y: i32,
    pad_value: u8,
) -> Result<(u64, Vec<i64>, i32, CudaPreprocessedImage), String> {
    let state = kernel_state()?;

    let expected_len = (scaled_width * scaled_height * 3) as usize;
    if bgr_u8.len() < expected_len {
        return Err(format!(
            "input buffer too small: got {}, need {}",
            bgr_u8.len(),
            expected_len
        ));
    }

    // Not wrapped in a keepalive resource -- purely Rust-scoped, freed
    // deterministically when this function returns, not subject to BEAM GC
    // timing at all. Only the *output* buffer needed the reuse treatment.
    let src_dev = state
        .stream
        .clone_htod(bgr_u8)
        .map_err(|e| format!("clone_htod failed: {e:?}"))?;

    let out_len = (3 * canvas_size * canvas_size) as usize;
    let pad_value_norm = (pad_value as f32) / 255.0f32;

    let mut output_guard = state
        .output
        .lock()
        .map_err(|_| "output buffer mutex poisoned".to_string())?;

    if !matches!(&*output_guard, Some((len, _)) if *len == out_len) {
        let fresh = state
            .stream
            .alloc_zeros::<f32>(out_len)
            .map_err(|e| format!("alloc_zeros failed: {e:?}"))?;
        *output_guard = Some((out_len, fresh));
    }

    let (_, dst_dev) = output_guard.as_mut().expect("just set above");

    unsafe {
        state
            .stream
            .launch_builder(&state.func)
            .arg(&src_dev)
            .arg(&mut *dst_dev)
            .arg(&scaled_width)
            .arg(&scaled_height)
            .arg(&canvas_size)
            .arg(&pad_x)
            .arg(&pad_y)
            .arg(&pad_value_norm)
            .launch(LaunchConfig::for_num_elems(
                (canvas_size * canvas_size) as u32,
            ))
    }
    .map_err(|e| format!("kernel launch failed: {e:?}"))?;

    // Ensure the kernel has actually finished writing before handing the raw
    // pointer off to onnxruntime -- run_cuda does no synchronization of its
    // own around externally-provided pointers, and src_dev must stay alive
    // until the kernel is done reading it (it's still in scope here).
    state
        .stream
        .synchronize()
        .map_err(|e| format!("stream synchronize failed: {e:?}"))?;

    let (raw_ptr, sync_guard) = dst_dev.device_ptr(&state.stream);
    drop(sync_guard);
    drop(output_guard);

    let shape = vec![1i64, 3, canvas_size as i64, canvas_size as i64];

    Ok((
        raw_ptr as u64,
        shape,
        state.ctx.ordinal() as i32,
        CudaPreprocessedImage,
    ))
}
