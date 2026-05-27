defmodule OrtexTest do
  use ExUnit.Case
  doctest Ortex

  test "resnet50" do
    0..500
    |> Enum.each(fn _ ->
      run()
    end)
  end

  defp run() do
    input = Nx.broadcast(0.0, {1, 3, 640, 640})
    # image = Evision.imread("C:/tmp/locator/Image01.jpg")

    # {:ok, scaled_image, scaler_config} =
    #  Ortex.Framework.YOLO.FrameScalers.fit(
    #    image,
    #    {640, 640},
    #    Ortex.Framework.YOLO.FrameScalers.Evision
    #  )

    # IO.inspect(Nx.shape(scaled_image))

    model =
      Ortex.load(
        "C:/Development/Source/Maskon/Elixir/flux_app/project/models/vax_locator_11n_25_640.onnx"
      )

    _ = Ortex.run(model, input)

    Process.sleep(100)
  end
end
