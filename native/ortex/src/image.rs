use image::{DynamicImage, GenericImage, ImageBuffer, Rgb, Rgba};
use rustler::{Binary, Encoder, Env, Error, NifResult, OwnedBinary, Term};
//use std::time::Instant;
use wide::{f32x16, u8x16}; // For optional profiling

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

pub fn prepare_image<'a>(
    env: Env<'a>,
    bin: Binary,
    width: u32,
    height: u32,
    size: u32,
) -> NifResult<Term<'a>> {
    let input_vec: Vec<u8> = bin.as_slice().to_vec();

    match resize_and_normalize_to_tensor(input_vec, width, height, size) {
        Ok(f32_bytes) => {
            // Create Binary via OwnedBinary for allocation and copy (handle Option safely)
            let mut owned = match OwnedBinary::new(f32_bytes.len()) {
                Some(o) => o,
                None => return Err(Error::BadArg.into()),
            };
            owned.as_mut_slice().copy_from_slice(&f32_bytes);
            let output_bin = Binary::from_owned(owned, env);

            Ok((atoms::ok(), output_bin).encode(env))
        }
        Err(e) => {
            // Create error Binary via OwnedBinary (handle Option safely)
            let mut owned = match OwnedBinary::new(e.len()) {
                Some(o) => o,
                None => return Err(Error::BadArg.into()),
            };
            owned.as_mut_slice().copy_from_slice(e.as_bytes());
            let err_bin = Binary::from_owned(owned, env);
            Ok((atoms::error(), err_bin).encode(env))
        }
    }
}

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
    let input_vec: Vec<u8> = bin.as_slice().to_vec();

    match normalize_bgr_to_padded_chw(
        input_vec,
        scaled_width,
        scaled_height,
        canvas_size,
        pad_x,
        pad_y,
        pad_value,
    ) {
        Ok(f32_bytes) => {
            let mut owned = match OwnedBinary::new(f32_bytes.len()) {
                Some(o) => o,
                None => return Err(Error::BadArg.into()),
            };
            owned.as_mut_slice().copy_from_slice(&f32_bytes);
            let output_bin = Binary::from_owned(owned, env);
            Ok((atoms::ok(), output_bin).encode(env))
        }
        Err(e) => {
            let mut owned = match OwnedBinary::new(e.len()) {
                Some(o) => o,
                None => return Err(Error::BadArg.into()),
            };
            owned.as_mut_slice().copy_from_slice(e.as_bytes());
            let err_bin = Binary::from_owned(owned, env);
            Ok((atoms::error(), err_bin).encode(env))
        }
    }
}

fn normalize_bgr_to_padded_chw(
    input: Vec<u8>,
    scaled_width: u32,
    scaled_height: u32,
    canvas_size: u32,
    pad_x: u32,
    pad_y: u32,
    pad_value: u8,
) -> Result<Vec<u8>, &'static str> {
    let (w, h) = (scaled_width as usize, scaled_height as usize);
    let canvas = canvas_size as usize;

    if input.len() < w * h * 3 {
        return Err("Input buffer smaller than scaled_width * scaled_height * 3");
    }

    let hw = canvas * canvas;
    let total_bytes = 4 * 3 * hw; // f32 bytes, batch=1, CHW
    let pad_norm = (pad_value as f32) / 255.0;
    let mut tensor: Vec<u8> = vec![0u8; total_bytes];

    // Fill with the (normalized) pad value first -- cheaper than branching
    // per-pixel inside the hot loop 
    {
        let pad_bytes = pad_norm.to_ne_bytes();
        for chunk in tensor.chunks_exact_mut(4) {
            chunk.copy_from_slice(&pad_bytes);
        }
    }

    let row_stride = w * 3;
    let norm_scale = f32x16::splat(1.0 / 255.0);

    // SIMD CHW build: 3 passes (R, G, B) -- reads BGR input, writes RGB
    // planes, at the (pad_x, pad_y) offset within the padded canvas.
    for bgr_ch in 0..3 {
        let rgb_idx = 2 - bgr_ch;
        for y in 0..h {
            let row_offset = y * row_stride;
            let canvas_row_base = bgr_ch * hw + (pad_y as usize + y) * canvas + pad_x as usize;
            let mut x = 0;
            while x + 16 <= w {
                let mut u8_arr = [0u8; 16];
                for i in 0..16 {
                    let pixel_offset = row_offset + (x + i) * 3 + rgb_idx;
                    u8_arr[i] = input[pixel_offset];
                }
                let u8_vals = u8x16::new(u8_arr);
                let f32_arr: [f32; 16] = u8_vals.as_array().map(|u| u as f32);
                let norm_vec = f32x16::new(f32_arr) * norm_scale;

                let tensor_offset = (canvas_row_base + x) * 4usize;
                unsafe {
                    let tensor_ptr = tensor.as_mut_ptr().add(tensor_offset).cast::<f32>();
                    let norm_arr = norm_vec.as_array();
                    for i in 0..16 {
                        *tensor_ptr.add(i) = norm_arr[i];
                    }
                }

                x += 16;
            }

            for tail in x..w {
                let pixel_offset = row_offset + tail * 3 + rgb_idx;
                let val_u8 = input[pixel_offset];
                let norm_val = (val_u8 as f32) / 255.0;
                let tensor_offset = (canvas_row_base + tail) * 4usize;
                unsafe {
                    let f32_ptr = tensor.as_mut_ptr().add(tensor_offset).cast::<f32>();
                    *f32_ptr = norm_val;
                }
            }
        }
    }

    Ok(tensor)
}

fn custom_letterbox_resize(
    original: &DynamicImage,
    target_w: u32,
    target_h: u32,
    pad_val: u8,
) -> DynamicImage {
    let orig_img = original.to_rgb8();
    let (orig_w, orig_h) = orig_img.dimensions();
    let orig_raw = orig_img.as_raw();

    let scale_w = target_w as f32 / orig_w as f32;
    let scale_h = target_h as f32 / orig_h as f32;
    let scale = scale_w.min(scale_h);

    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    // Resize to new size (nearest)
    let mut resized_raw = Vec::with_capacity((new_w * new_h * 3) as usize);
    resized_raw.resize((new_w * new_h * 3) as usize, 0u8);

    let scale_x = orig_w as f32 / new_w as f32;
    let scale_y = orig_h as f32 / new_h as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = (x as f32 * scale_x).round() as usize;
            let src_y = (y as f32 * scale_y).round() as usize;

            let src_x = src_x.min((orig_w - 1) as usize);
            let src_y = src_y.min((orig_h - 1) as usize);

            let src_offset = ((src_y * orig_w as usize + src_x) * 3) as usize;
            let dst_offset = ((y * new_w + x) * 3) as usize;

            resized_raw[dst_offset] = orig_raw[src_offset]; // R
            resized_raw[dst_offset + 1] = orig_raw[src_offset + 1]; // G
            resized_raw[dst_offset + 2] = orig_raw[src_offset + 2]; // B
        }
    }

    let resized_img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(new_w, new_h, resized_raw).unwrap();

    // Pad to target (center, with pad_val)
    let pad_x = ((target_w - new_w) / 2) as usize;
    let pad_y = ((target_h - new_h) / 2) as usize;

    let mut padded_raw = Vec::with_capacity((target_w * target_h * 3) as usize);
    padded_raw.resize((target_w * target_h * 3) as usize, pad_val);

    let padded_img = ImageBuffer::from_raw(target_w, target_h, padded_raw).unwrap();
    let mut padded = DynamicImage::ImageRgb8(padded_img);

    // Copy resized to center (convert Rgb<u8> to Rgba<u8> for put_pixel)
    for y in 0..new_h {
        for x in 0..new_w {
            if let Some(px) = resized_img.get_pixel_checked(x, y) {
                let rgba_px = Rgba([px.0[0], px.0[1], px.0[2], 255u8]);
                padded.put_pixel(pad_x as u32 + x, pad_y as u32 + y, rgba_px);
            }
        }
    }

    //let _ = padded.save("C:/tmp/test.jpg");

    padded
}

fn resize_and_normalize_to_tensor(
    input: Vec<u8>,
    width: u32,
    height: u32,
    size: u32,
) -> Result<Vec<u8>, &'static str> {
    //let start_load = Instant::now();
    let original_img = if width == 0 && height == 0 {
        // File format (JPEG/BMP/etc.)
        image::load_from_memory(&input)
            .map_err(|_| "Failed to load image")?
            .to_rgb8()
    } else {
        ImageBuffer::from_raw(width, height, input).ok_or("Invalid raw dimensions")?
    };
    //println!("Load time: {:?}", start_load.elapsed());

    //let start_resize = Instant::now();
    let img = {
        let dynamic_img = DynamicImage::ImageRgb8(original_img);
        custom_letterbox_resize(&dynamic_img, size, size, 114) // Custom fast resize
    };
    // println!("Resize time: {:?}", start_resize.elapsed());

    //let start_loop = Instant::now();
    let h = size as usize;
    let w = h;

    let hw = h * w;
    let total_bytes = 4 * 1 * 3 * hw; // f32 bytes, batch=1, CHW
    let mut tensor: Vec<u8> = vec![0u8; total_bytes]; // Pre-allocate full len with 0s

    let rgb_img = img.to_rgb8(); // Bind temporary to extend lifetime
    let raw_pixels = rgb_img.as_raw(); // Now safe borrow
    let bytes_per_pixel = 3usize;
    let row_stride = w * bytes_per_pixel; // Hoist row calculation
    let norm_scale = f32x16::splat(1.0 / 255.0); // SIMD scale

    // SIMD CHW build: 3 passes (B, G, R for BGR)
    for bgr_ch in 0..3 {
        let rgb_idx = 2 - bgr_ch; // BGR mapping
        for y in 0..h {
            let row_offset = y * row_stride;
            let mut x = 0;
            while x + 16 <= w {
                // Load 16 u8 for channel (strided)
                let mut u8_arr = [0u8; 16];
                for i in 0..16 {
                    let pixel_offset = row_offset + (x + i) * bytes_per_pixel + rgb_idx;
                    u8_arr[i] = raw_pixels[pixel_offset];
                }
                let u8_vals = u8x16::new(u8_arr);

                // Cast/normalize SIMD
                let f32_arr: [f32; 16] = u8_vals.as_array().map(|u| u as f32);
                let f32_vals = f32x16::new(f32_arr);
                let norm_vec = f32_vals * norm_scale;

                // Store via ptr loop (unsafe wrapped)
                let tensor_base = bgr_ch * hw + y * w + x;
                let tensor_offset = tensor_base * 4usize; // f32 stride
                unsafe {
                    let tensor_ptr = tensor.as_mut_ptr().add(tensor_offset).cast::<f32>();
                    let norm_arr = norm_vec.as_array();
                    for i in 0..16 {
                        let f32_ptr = tensor_ptr.add(i);
                        *f32_ptr = norm_arr[i];
                    }
                }

                x += 16;
            }

            // Scalar tail
            for tail in x..w {
                let pixel_offset = row_offset + tail * bytes_per_pixel + rgb_idx;
                let val_u8 = raw_pixels[pixel_offset];
                let norm_val = (val_u8 as f32) * norm_scale.as_array()[0];
                let tensor_offset = (bgr_ch * hw + y * w + tail) * 4usize;
                unsafe {
                    let f32_ptr = tensor.as_mut_ptr().add(tensor_offset).cast::<f32>();
                    *f32_ptr = norm_val;
                }
            }
        }
    }

    // println!("Loop time: {:?}", start_loop.elapsed());

    Ok(tensor)
}
