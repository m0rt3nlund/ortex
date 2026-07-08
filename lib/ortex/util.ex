defmodule Ortex.Util do
  @moduledoc false
  @doc """
    Copies the libraries downloaded during the ORT build into a path that
    Elixir can use
  """
  def copy_ort_libs() do
    # __DIR__ always points at this file's real location on disk, even when
    # ortex is used as a path dependency and :code.priv_dir(:ortex) resolves
    # through a _build symlink that doesn't contain a native/ dir at all.
    crate_root = Path.expand("../../native/ortex", __DIR__)

    rust_env =
      case Path.join([crate_root, "target/release"]) |> File.ls() do
        {:ok, _} -> "release"
        _ -> "debug"
      end

    # ort 2.x extracts the library into build/<hash>/out/ subdirectories,
    # so search recursively with ** instead of a flat glob.
    rust_path = Path.join([crate_root, "target", rust_env])

    pattern =
      case :os.type() do
        {:win32, _} -> Path.join([rust_path, "**", "libonnxruntime*.dll*"])
        {:unix, :darwin} -> Path.join([rust_path, "**", "libonnxruntime*.dylib*"])
        {:unix, _} -> Path.join([rust_path, "**", "libonnxruntime*.so*"])
      end

    onnx_runtime_paths =
      pattern
      |> Path.wildcard()
      |> Enum.reject(&File.dir?/1)
      |> Enum.uniq_by(&Path.basename/1)

    destination_dir = Path.join([:code.priv_dir(:ortex), "native"])

    Enum.each(onnx_runtime_paths, fn src ->
      File.cp!(src, Path.join(destination_dir, Path.basename(src)))
    end)
  end
end
