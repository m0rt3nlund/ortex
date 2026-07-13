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
end
