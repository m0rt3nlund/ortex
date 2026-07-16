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

  @enforce_keys [:reference, :inputs, :outputs]
  defstruct [:reference, :inputs, :outputs]

  @doc false
  def load(path, eps \\ [:cpu], opt \\ 3) do
    normalized =
      Enum.map(eps, fn
        ep when is_atom(ep) ->
          {ep, []}

        {ep, opts} ->
          {ep, Enum.map(opts, fn {k, v} -> {Atom.to_string(k), to_string(v)} end)}
      end)

    case Ortex.Native.init(path, normalized, opt) do
      {:error, msg} ->
        raise msg

      model ->
        {inputs, outputs} =
          Ortex.Native.show_session(model)
          |> case do
            {inputs, outputs} ->
              inputs =
                inputs
                |> Enum.reduce(%{}, fn {id, _, shape}, acc -> Map.put(acc, id, shape) end)

              outputs =
                outputs
                |> Enum.reduce(%{}, fn {id, _, shape}, acc -> Map.put(acc, id, shape) end)

              {inputs, outputs}

            _ ->
              {[], []}
          end

        %Ortex.Model{reference: model, inputs: inputs, outputs: outputs}
    end
  end

  # A pre-built raw CUDA pointer
  @doc false
  def run(%Ortex.Model{reference: model}, %Ortex.CudaTensor{} = cuda_tensor) do
    result =
      Ortex.Native.run_cuda(model, [
        {cuda_tensor.ptr, cuda_tensor.shape, cuda_tensor.dtype_str, cuda_tensor.dtype_bits,
         cuda_tensor.device_index}
      ])

    keepalive = cuda_tensor.keepalive
    _ = keepalive

    pack_output(result)
  end

  @doc false
  def run(%Ortex.Model{} = model, tensor) when not is_tuple(tensor) do
    run(model, {tensor})
  end

  @doc false
  def run(%Ortex.Model{reference: model}, tensors) do
    tensor_list = Tuple.to_list(tensors)

    raw_output =
      if Enum.all?(tensor_list, &cuda_backed?/1) do
        {cuda_inputs, keepalives} = tensor_list |> Enum.map(&cuda_input/1) |> Enum.unzip()

        result = Ortex.Native.run_cuda(model, cuda_inputs)

        _ = keepalives
        result
      else
        inputs =
          Enum.map(tensor_list, fn %Nx.Tensor{shape: shape, type: {type_atom, bits}} = tensor ->
            {Nx.to_binary(tensor), Tuple.to_list(shape), Atom.to_string(type_atom), bits}
          end)

        Ortex.Native.run_binary(model, inputs)
      end

    pack_output(raw_output)
  end

  # Pack raw_output ({ref, shape, dtype_atom, dtype_bits} tuples
  defp pack_output(raw_output) do
    output =
      case raw_output do
        {:error, msg} -> raise msg
        output -> output
      end

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

  # Only take the zero-copy path for tensors already GPU-resident via Torchx
  defp cuda_backed?(%Nx.Tensor{data: %{__struct__: Torchx.Backend, ref: {:cuda, _}}}), do: true
  defp cuda_backed?(_), do: false

  # Returns `{{ptr, shape, dtype_str, dtype_bits, device_index}, keepalive}`
  defp cuda_input(%Nx.Tensor{data: %{ref: ref}, shape: shape, type: {type_atom, bits}}) do
    {ptr, _shape, _dtype, {_device_type, device_index}, keepalive} = Torchx.data_ptr(ref)
    {{ptr, Tuple.to_list(shape), Atom.to_string(type_atom), bits, device_index}, keepalive}
  end

  def create_mask(coefficients, mask_prototypes, threshold \\ 0.5) do
    prototypes_bin = Nx.to_binary(mask_prototypes)
    # Tuple like {1, 32, 240, 240}
    {_, _, width, height} = proto_shape = Nx.shape(mask_prototypes)
    {_, dtype_bits} = Nx.type(mask_prototypes)

    Ortex.Native.create_mask(coefficients, prototypes_bin, proto_shape, dtype_bits, threshold)
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
