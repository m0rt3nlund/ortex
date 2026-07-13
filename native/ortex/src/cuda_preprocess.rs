//! Custom CUDA kernel for the "rest of fit_rust" preprocessing step: BGR->RGB
//! swap, u8->f32 normalize, HWC->CHW transpose, and letterbox padding, fused
//! into one GPU kernel. 

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::{Arc, OnceLock};

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
}

// OnceLock rather than re-compiling/re-loading per call
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
            Ok(KernelState { ctx, stream, func })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Holds the GPU output buffer from the preprocessing kernel
pub struct CudaPreprocessedImage {
    #[allow(dead_code)]
    pub buf: CudaSlice<f32>,
}

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

    let src_dev = state
        .stream
        .clone_htod(bgr_u8)
        .map_err(|e| format!("clone_htod failed: {e:?}"))?;

    let out_len = (3 * canvas_size * canvas_size) as usize;
    let mut dst_dev: CudaSlice<f32> = state
        .stream
        .alloc_zeros(out_len)
        .map_err(|e| format!("alloc_zeros failed: {e:?}"))?;

    let pad_value_norm = (pad_value as f32) / 255.0f32;

    unsafe {
        state
            .stream
            .launch_builder(&state.func)
            .arg(&src_dev)
            .arg(&mut dst_dev)
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

    // Ensure the kernel has actually finished writing before handing the raw pointer off to onnxruntime
    state
        .stream
        .synchronize()
        .map_err(|e| format!("stream synchronize failed: {e:?}"))?;

    let (raw_ptr, sync_guard) = dst_dev.device_ptr(&state.stream);
    drop(sync_guard);

    let shape = vec![1i64, 3, canvas_size as i64, canvas_size as i64];

    Ok((
        raw_ptr as u64,
        shape,
        state.ctx.ordinal() as i32,
        CudaPreprocessedImage { buf: dst_dev },
    ))
}
