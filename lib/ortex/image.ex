defmodule Ortex.Image do
  def prepare(image, width, height, size \\ 640) do
    Ortex.Native.prepare_image(image, width, height, size)
  end
end
