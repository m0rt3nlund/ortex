defmodule Ortex.Model do
  @moduledoc """
  A model for running Ortex inference with.

  Implements a human-readable representation of a model including the name, dimension, and
  type of each input and output

  ```
  #Ortex.Model<
  inputs: [{"x", "Int32", [nil, 100]}, {"y", "Float32", [nil, 100]}]
  outputs: [
    {"9", "Float32", [nil, 10]},
    {"onnx::Add_7", "Float32", [nil, 10]},
    {"onnx::Add_8", "Float32", [nil, 10]}
  ]>
  ```

  `nil` values represent dynamic dimensions
  """

  @enforce_keys [:reference]
  defstruct [:reference]

  @doc false
  def load(path, eps \\ [:cpu], opt \\ 3) do
    case Ortex.Native.init(path, eps, opt) do
      {:error, msg} ->
        raise msg

      model ->
        %Ortex.Model{reference: model}
    end
  end

  @doc false
  def run(%Ortex.Model{} = model, tensor) when not is_tuple(tensor) do
    run(model, {tensor})
  end

  @doc false
  def run(%Ortex.Model{reference: model}, tensors) do
    {t_transfer, tensor_refs} = :timer.tc(fn ->
      tensors
      |> Tuple.to_list()
      |> Enum.map(fn x -> x |> Nx.backend_transfer(Ortex.Backend) end)
      |> Enum.map(fn %Nx.Tensor{data: %Ortex.Backend{ref: x}} -> x end)
    end)

    {t_nif, raw_output} = :timer.tc(fn -> Ortex.Native.run(model, tensor_refs) end)

    IO.puts(:stderr, "[ortex elixir] backend_transfer=#{t_transfer}µs  nif_run=#{t_nif}µs")

    output =
      case raw_output do
        {:error, msg} -> raise msg
        output -> output
      end

    # Pack the output into new Ortex.Backend tensor(s)
    output
    |> Enum.map(fn {ref, shape, dtype_atom, dtype_bits} ->
      %Nx.Tensor{
        data: %Ortex.Backend{ref: ref},
        shape: shape |> List.to_tuple(),
        type: {dtype_atom, dtype_bits},
        names: List.duplicate(nil, length(shape))
      }
    end)
    |> List.to_tuple()
  end

  def create_mask(coefficients, mask_prototypes, threshold \\ 0.5) do
    prototypes_bin = Nx.to_binary(mask_prototypes)
    # Tuple like {1, 32, 240, 240}
    {_, _, width, height} = proto_shape = Nx.shape(mask_prototypes)

    Ortex.Native.create_mask(coefficients, prototypes_bin, proto_shape, threshold)
    |> case do
      binary_mask when is_binary(binary_mask) ->
        binary_mask
        |> Nx.from_binary(:u8)
        |> Nx.reshape({width, height})

      error ->
        error
    end
  end
end

defimpl Inspect, for: Ortex.Model do
  import Inspect.Algebra

  def inspect(%Ortex.Model{reference: model}, inspect_opts) do
    case Ortex.Native.show_session(model) do
      {:error, msg} ->
        raise msg

      {inputs, outputs} ->
        force_unfit(
          concat([
            color("#Ortex.Model<", :map, inspect_opts),
            line(),
            nest(concat(["  inputs: ", Inspect.List.inspect(inputs, inspect_opts)]), 2),
            line(),
            nest(concat(["  outputs: ", Inspect.List.inspect(outputs, inspect_opts)]), 2),
            color(">", :map, inspect_opts)
          ])
        )
    end
  end
end
