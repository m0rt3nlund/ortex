defmodule Ortex.Image do
  def prepare(image, width, height, size \\ 640) do
    Ortex.Native.prepare_image(image, width, height, size)
  end


  def prepare_resized(image, scaled_width, scaled_height, canvas_size, pad_x, pad_y, pad_value) do
    Ortex.Native.prepare_resized_image(
      image,
      scaled_width,
      scaled_height,
      canvas_size,
      pad_x,
      pad_y,
      pad_value
    )
  end

  @doc "Like `prepare_resized/7`, but does normalize/BGR->RGB/HWC->CHW/pad on GPU via a custom CUDA kernel."
  def prepare_resized_cuda(
        image,
        scaled_width,
        scaled_height,
        canvas_width,
        canvas_height,
        pad_x,
        pad_y,
        pad_value,
        half \\ false
      ) do
    {ptr, shape, device_index, dtype_bits, keepalive} =
      Ortex.Native.prepare_resized_image_cuda(
        image,
        scaled_width,
        scaled_height,
        canvas_width,
        canvas_height,
        pad_x,
        pad_y,
        pad_value,
        half
      )

    {:ok,
     %Ortex.CudaTensor{
       ptr: ptr,
       shape: shape,
       dtype_str: "f",
       dtype_bits: dtype_bits,
       device_index: device_index,
       keepalive: keepalive
     }}
  end
end
