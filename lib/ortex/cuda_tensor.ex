defmodule Ortex.CudaTensor do
  @moduledoc """
  Wraps a raw CUDA device pointer produced directly by a native preprocessing
  kernel for zero-copy handoff straight into `Ortex.run/2` without ever going
  through Nx or Torchx.

  `keepalive` must be kept referenced by the caller for as long as `ptr` is in use
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
