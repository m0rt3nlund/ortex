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

  @doc false
  def unload(%Ortex.Model{reference: model}), do: Ortex.Native.unload(model)

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

    inputs =
      Enum.map(tensor_list, fn %Nx.Tensor{shape: shape, type: {type_atom, bits}} = tensor ->
        {Nx.to_binary(tensor), Tuple.to_list(shape), Atom.to_string(type_atom), bits}
      end)

    Ortex.Native.run_binary(model, inputs)
    |> pack_output()
  end

  # Like `run/2`, but for a raw CUDA pointer input (`%Ortex.CudaTensor{}`),
  # additionally pins each output named in `cuda_output_names` to CUDA device
  # memory via IoBinding instead of letting ONNX Runtime copy it to the host.

  @doc false
  def run_pinned(%Ortex.Model{reference: model}, %Ortex.CudaTensor{} = cuda_tensor, cuda_output_names) do
    {host_outputs, cuda_outputs} =
      Ortex.Native.run_cuda_pinned(
        model,
        [
          {cuda_tensor.ptr, cuda_tensor.shape, cuda_tensor.dtype_str, cuda_tensor.dtype_bits,
           cuda_tensor.device_index}
        ],
        cuda_output_names,
        cuda_tensor.device_index
      )

    keepalive = cuda_tensor.keepalive
    _ = keepalive

    pack_pinned_output(host_outputs, cuda_outputs)
  end

  defp pack_pinned_output(host_outputs, cuda_outputs) do
    host_map =
      Map.new(host_outputs, fn {name, ref, shape, dtype_atom, dtype_bits} ->
        {name,
         %Nx.Tensor{
           data: %Ortex.Backend{ref: ref},
           shape: shape |> List.to_tuple(),
           type: {dtype_atom, dtype_bits},
           names: List.duplicate(nil, length(shape))
         }}
      end)

    cuda_map =
      Map.new(cuda_outputs, fn {name, ptr, shape, dtype_str, dtype_bits, device_index, keepalive} ->
        {name,
         %Ortex.CudaOutput{
           ptr: ptr,
           shape: shape,
           dtype_str: dtype_str,
           dtype_bits: dtype_bits,
           device_index: device_index,
           keepalive: keepalive
         }}
      end)

    Map.merge(host_map, cuda_map)
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
