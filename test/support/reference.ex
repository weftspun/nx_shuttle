defmodule NxShuttle.Reference do
  @moduledoc false

  # Chooses what the lowering is measured against, and says which one it chose.
  #
  # THE TWO ENGINES DO NOT AGREE, which is the whole reason this module names its choice
  # instead of quietly picking one. A float32 -> int32 -> float32 round trip agreed with Nx
  # under ONNX Runtime and disagreed by 0.875 under the Dataflow Compiler, which accepted the
  # graph and then did not perform the cast. So a run against the fallback is a strictly weaker
  # statement than a run against the compiler, and reporting both as "passed" would be the
  # silent-skip failure wearing a different hat.
  #
  #   :dfc    the compiler that has to accept the graph. Authoritative. Needs a licensed wheel
  #           and a multi-gigabyte image, so it is exactly the reference that goes missing.
  #   :ortex  ONNX Runtime in-process. Says the bytes are well formed and the arithmetic is
  #           what Nx meant. Says NOTHING about whether the accelerator toolchain accepts it.

  alias NxShuttle.DFC

  @doc """
  Returns `{:ok, engine}` for the best available reference, or `{:error, reasons}`.
  """
  def engine do
    case DFC.available() do
      :ok ->
        {:ok, :dfc}

      {:error, dfc_why} ->
        case ortex_available() do
          :ok -> {:ok, {:ortex, dfc_why}}
          {:error, ortex_why} -> {:error, {dfc_why, ortex_why}}
        end
    end
  end

  @doc """
  Runs `cases` through `engine`. Each case is `%{name:, model:, input:}`.

  Returns a map of name to `{:ok, tensor}` or `{:error, reason}`, in Nx's own layout.
  """
  def run(:dfc, cases), do: DFC.run(cases)

  def run({:ortex, _}, cases) do
    Map.new(cases, fn %{name: name, model: model, input: input} ->
      path = Path.join(System.tmp_dir!(), "#{name}_#{System.unique_integer([:positive])}.onnx")
      File.write!(path, model)

      try do
        session = Ortex.load(path)
        {out} = Ortex.run(session, {input})
        {name, {:ok, Nx.backend_transfer(out, Nx.BinaryBackend)}}
      rescue
        e -> {name, {:error, Exception.message(e)}}
      after
        File.rm(path)
      end
    end)
  end

  @doc """
  A one-line description of what a green run actually proved.
  """
  def caveat(:dfc), do: "measured against the Dataflow Compiler's own emulator (authoritative)"

  def caveat({:ortex, why}),
    do:
      "FALLBACK: measured against ONNX Runtime only, because the compiler is unavailable " <>
        "(#{why}). This proves the arithmetic, NOT that the toolchain accepts the graph."

  # Ortex is a Rustler NIF and `Ortex.load/1` ABORTS THE VM on windows-x86_64 rather than
  # returning an error, so it cannot be probed by trying it -- a failed probe would take the
  # test run with it. The platform is checked instead, which is a coarser test and a safe one.
  defp ortex_available do
    case :os.type() do
      {:win32, _} ->
        {:error, "ortex aborts the BEAM on windows-x86_64, so it cannot stand in here"}

      _ ->
        if Code.ensure_loaded?(Ortex),
          do: :ok,
          else: {:error, "ortex is not compiled into this environment"}
    end
  end
end
