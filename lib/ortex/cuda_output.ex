defmodule Ortex.CudaOutput do
  @moduledoc """
  Wraps a raw CUDA device pointer for a model output that ONNX Runtime was told
  (via IoBinding) to leave resident on the device, instead of copying it to the
  host the way an ordinary `Ortex.run/2` output is. Produced by
  `Ortex.Model.run_pinned/3`, the mirror image of `Ortex.CudaTensor`,
  this one wraps a device pointer produced by ONNX Runtime itself, for zero-copy
  handoff *out of* it.

  `keepalive` must be kept referenced by the caller for as long as `ptr` is in
  use, it owns the underlying ONNX Runtime `Value`, so the device buffer is
  freed once nothing references it anymore, and not before.
  """
  defstruct [:ptr, :shape, :dtype_str, :dtype_bits, :device_index, :keepalive]

  @type t :: %__MODULE__{
          ptr: non_neg_integer(),
          shape: [integer()],
          dtype_str: String.t(),
          dtype_bits: non_neg_integer(),
          device_index: integer(),
          keepalive: reference()
        }
end
