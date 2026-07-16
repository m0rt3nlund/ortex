//! Custom CUDA kernel for the "rest of fit_rust" preprocessing step: BGR->RGB
//! swap, u8->f32/f16 normalize, HWC->CHW transpose, and letterbox padding,
//! fused into one GPU kernel. Fed by the same CPU-side Evision resize the
//! Torchx path already uses (resize itself stays on CPU deliberately -- see
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
//! moment). Reusing one buffer removes that dependency entirely. Since f32
//! and f16 output are mutually exclusive per call but both need to stay
//! reusable across calls, each dtype gets its own persistent buffer slot
//! rather than sharing one via an unsafe transmute.
//!
//! Safety note: reusing a single buffer per dtype (rather than a small pool)
//! is only correct because the pipeline that calls this is currently
//! strictly sequential -- frame N+1's preprocessing never starts until frame
//! N's entire Inferencer.process/3 (preprocess -> Ortex.run -> postprocess)
//! has returned, so by the time this function's Mutex is next locked, the
//! previous frame's TensorRT read of this same buffer has already finished.
//! If the pipeline is ever changed to overlap frames (e.g. splitting
//! preprocess/infer into separate async GenStage stages), this needs to
//! become an actual pool (2-3 buffers, round-robin/refcounted) instead --
//! a single shared buffer would then be a genuine data race.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use half::f16;
use std::sync::{Arc, Mutex, OnceLock};

// `cuda_fp16.h` is one of the handful of standard CUDA headers NVRTC bundles
// internally for JIT use (no toolkit include path needs to be configured),
// specifically to support half-precision kernels like this one.
const KERNEL_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void bgr_to_padded_rgb_chw_f32(
    const unsigned char* src,
    float* dst,
    int scaled_width,
    int scaled_height,
    int canvas_width,
    int canvas_height,
    int pad_x,
    int pad_y,
    float pad_value_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = canvas_width * canvas_height;
    if (idx >= total) return;

    int cy = idx / canvas_width;
    int cx = idx % canvas_width;
    int hw = canvas_width * canvas_height;

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

extern "C" __global__ void bgr_to_padded_rgb_chw_f16(
    const unsigned char* src,
    __half* dst,
    int scaled_width,
    int scaled_height,
    int canvas_width,
    int canvas_height,
    int pad_x,
    int pad_y,
    float pad_value_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = canvas_width * canvas_height;
    if (idx >= total) return;

    int cy = idx / canvas_width;
    int cx = idx % canvas_width;
    int hw = canvas_width * canvas_height;

    int sx = cx - pad_x;
    int sy = cy - pad_y;

    if (sx >= 0 && sx < scaled_width && sy >= 0 && sy < scaled_height) {
        int src_offset = (sy * scaled_width + sx) * 3;
        unsigned char b = src[src_offset];
        unsigned char g = src[src_offset + 1];
        unsigned char r = src[src_offset + 2];

        dst[0 * hw + idx] = __float2half_rn(((float)r) / 255.0f);
        dst[1 * hw + idx] = __float2half_rn(((float)g) / 255.0f);
        dst[2 * hw + idx] = __float2half_rn(((float)b) / 255.0f);
    } else {
        __half pad = __float2half_rn(pad_value_norm);
        dst[0 * hw + idx] = pad;
        dst[1 * hw + idx] = pad;
        dst[2 * hw + idx] = pad;
    }
}
"#;

struct KernelState {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func_f32: CudaFunction,
    func_f16: CudaFunction,
    output_f32: Mutex<Option<(usize, CudaSlice<f32>)>>,
    output_f16: Mutex<Option<(usize, CudaSlice<f16>)>>,
}

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
            let func_f32 = module
                .load_function("bgr_to_padded_rgb_chw_f32")
                .map_err(|e| format!("load_function (f32) failed: {e:?}"))?;
            let func_f16 = module
                .load_function("bgr_to_padded_rgb_chw_f16")
                .map_err(|e| format!("load_function (f16) failed: {e:?}"))?;
            Ok(KernelState {
                ctx,
                stream,
                func_f32,
                func_f16,
                output_f32: Mutex::new(None),
                output_f16: Mutex::new(None),
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

pub struct CudaPreprocessedImage;

#[allow(clippy::too_many_arguments)]
fn launch_f32(
    state: &KernelState,
    src_dev: &CudaSlice<u8>,
    out_len: usize,
    scaled_width: i32,
    scaled_height: i32,
    canvas_width: i32,
    canvas_height: i32,
    pad_x: i32,
    pad_y: i32,
    pad_value_norm: f32,
) -> Result<u64, String> {
    let mut output_guard = state
        .output_f32
        .lock()
        .map_err(|_| "f32 output buffer mutex poisoned".to_string())?;

    if !matches!(&*output_guard, Some((len, _)) if *len == out_len) {
        let fresh = state
            .stream
            .alloc_zeros::<f32>(out_len)
            .map_err(|e| format!("alloc_zeros (f32) failed: {e:?}"))?;
        *output_guard = Some((out_len, fresh));
    }

    let (_, dst_dev) = output_guard.as_mut().expect("just set above");

    unsafe {
        state
            .stream
            .launch_builder(&state.func_f32)
            .arg(src_dev)
            .arg(&mut *dst_dev)
            .arg(&scaled_width)
            .arg(&scaled_height)
            .arg(&canvas_width)
            .arg(&canvas_height)
            .arg(&pad_x)
            .arg(&pad_y)
            .arg(&pad_value_norm)
            .launch(LaunchConfig::for_num_elems(
                (canvas_width * canvas_height) as u32,
            ))
    }
    .map_err(|e| format!("kernel launch (f32) failed: {e:?}"))?;

    state
        .stream
        .synchronize()
        .map_err(|e| format!("stream synchronize failed: {e:?}"))?;

    let (raw_ptr, sync_guard) = dst_dev.device_ptr(&state.stream);
    drop(sync_guard);
    drop(output_guard);

    Ok(raw_ptr as u64)
}

#[allow(clippy::too_many_arguments)]
fn launch_f16(
    state: &KernelState,
    src_dev: &CudaSlice<u8>,
    out_len: usize,
    scaled_width: i32,
    scaled_height: i32,
    canvas_width: i32,
    canvas_height: i32,
    pad_x: i32,
    pad_y: i32,
    pad_value_norm: f32,
) -> Result<u64, String> {
    let mut output_guard = state
        .output_f16
        .lock()
        .map_err(|_| "f16 output buffer mutex poisoned".to_string())?;

    if !matches!(&*output_guard, Some((len, _)) if *len == out_len) {
        let fresh = state
            .stream
            .alloc_zeros::<f16>(out_len)
            .map_err(|e| format!("alloc_zeros (f16) failed: {e:?}"))?;
        *output_guard = Some((out_len, fresh));
    }

    let (_, dst_dev) = output_guard.as_mut().expect("just set above");

    unsafe {
        state
            .stream
            .launch_builder(&state.func_f16)
            .arg(src_dev)
            .arg(&mut *dst_dev)
            .arg(&scaled_width)
            .arg(&scaled_height)
            .arg(&canvas_width)
            .arg(&canvas_height)
            .arg(&pad_x)
            .arg(&pad_y)
            .arg(&pad_value_norm)
            .launch(LaunchConfig::for_num_elems(
                (canvas_width * canvas_height) as u32,
            ))
    }
    .map_err(|e| format!("kernel launch (f16) failed: {e:?}"))?;

    state
        .stream
        .synchronize()
        .map_err(|e| format!("stream synchronize failed: {e:?}"))?;

    let (raw_ptr, sync_guard) = dst_dev.device_ptr(&state.stream);
    drop(sync_guard);
    drop(output_guard);

    Ok(raw_ptr as u64)
}

/// Returns (raw_device_ptr, shape [1,3,canvas_height,canvas_width],
/// device_ordinal, dtype_bits (32 or 16), keepalive).
#[allow(clippy::too_many_arguments)]
pub fn prepare_resized_bgr_cuda(
    bgr_u8: &[u8],
    scaled_width: i32,
    scaled_height: i32,
    canvas_width: i32,
    canvas_height: i32,
    pad_x: i32,
    pad_y: i32,
    pad_value: u8,
    half: bool,
) -> Result<(u64, Vec<i64>, i32, usize, CudaPreprocessedImage), String> {
    let state = kernel_state()?;

    let expected_len = (scaled_width * scaled_height * 3) as usize;
    if bgr_u8.len() < expected_len {
        return Err(format!(
            "input buffer too small: got {}, need {}",
            bgr_u8.len(),
            expected_len
        ));
    }

    // Not wrapped in a keepalive resource 
    let src_dev = state
        .stream
        .clone_htod(bgr_u8)
        .map_err(|e| format!("clone_htod failed: {e:?}"))?;

    let out_len = (3 * canvas_width * canvas_height) as usize;
    let pad_value_norm = (pad_value as f32) / 255.0f32;

    let (raw_ptr, dtype_bits) = if half {
        let ptr = launch_f16(
            state,
            &src_dev,
            out_len,
            scaled_width,
            scaled_height,
            canvas_width,
            canvas_height,
            pad_x,
            pad_y,
            pad_value_norm,
        )?;
        (ptr, 16usize)
    } else {
        let ptr = launch_f32(
            state,
            &src_dev,
            out_len,
            scaled_width,
            scaled_height,
            canvas_width,
            canvas_height,
            pad_x,
            pad_y,
            pad_value_norm,
        )?;
        (ptr, 32usize)
    };

    // NCHW: (batch, channel, height, width)
    let shape = vec![1i64, 3, canvas_height as i64, canvas_width as i64];

    Ok((
        raw_ptr,
        shape,
        state.ctx.ordinal() as i32,
        dtype_bits,
        CudaPreprocessedImage,
    ))
}
