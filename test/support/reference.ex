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
  Every engine that can run here, and a reason for each that cannot.

  Both are DEPLOYMENT targets, so this returns all of them rather than picking one. A defect
  that only shows up as a disagreement between them -- the cast above -- is invisible to a
  check that runs whichever happened to be available.
  """
  def engines do
    checks = [{:dfc, DFC.available()}, {:ortex, ortex_available()}]
    {ok, missing} = Enum.split_with(checks, fn {_e, r} -> r == :ok end)
    {Enum.map(ok, &elem(&1, 0)), Enum.map(missing, fn {e, {:error, w}} -> {e, w} end)}
  end

  @doc """
  Runs `cases` through `engine`. Each case is `%{name:, model:, input:}`.

  Returns a map of name to `{:ok, tensor}` or `{:error, reason}`, in Nx's own layout.
  """
  def run(:dfc, cases), do: DFC.run(cases)

  def run(:ortex, cases), do: run({:ortex, nil}, cases)

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
  What a green run actually proved, given which engines ran.
  """
  def caveat([:dfc, :ortex]),
    do: "both deployment targets ran and were cross-checked against each other"

  def caveat([:dfc]),
    do:
      "PRIMARY ONLY: the accelerator toolchain ran. Nothing was cross-checked, so a graph " <>
        "that deploys differently on the backup runtime would not show here."

  def caveat([:ortex]),
    do:
      "BACKUP ONLY: ONNX Runtime ran. This proves the arithmetic, NOT that the accelerator " <>
        "toolchain accepts the graph."

  def caveat([]), do: "NOTHING RAN"

  # RETRACTED: an earlier version refused ortex on windows-x86_64, claiming `Ortex.load/1`
  # aborts the VM. That was ortex 0.1.7, which links ONNX Runtime DYNAMICALLY and died when
  # `onnxruntime.dll` was not beside the NIF. 0.1.10 links it statically -- one 18.5 MB
  # `libortex.dll` and no separate library -- and loads and runs here, max|diff| 0.0. The
  # platform check is gone; availability is decided by whether the module is there.
  defp ortex_available do
    if Code.ensure_loaded?(Ortex),
      do: :ok,
      else: {:error, "ortex is not compiled into this environment"}
  end

  @doc """
  The highest ONNX opset each engine can load, measured rather than declared.

  Ortex bundles its own ONNX Runtime through the `ort` crate, so its ceiling is fixed at build
  time and cannot be raised from mix.exs. Measured on 0.1.10: 21 loads, 22 does not. The
  compiler parsed 23. The emitter defaults to 17, which is under both, so nothing is currently
  blocked -- but a graph needing 22 would deploy to the accelerator and not to the backup, and
  that is the kind of thing that should fail a check rather than surface in production.
  """
  def opset_ceiling(:ortex), do: 21
  def opset_ceiling(:dfc), do: 23
end
