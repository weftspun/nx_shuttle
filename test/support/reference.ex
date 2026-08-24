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
  #   :dfc      the compiler that has to accept the graph. Authoritative. Needs a licensed wheel
  #             and a multi-gigabyte image, so it is exactly the reference that goes missing.
  #   :specref  the ONNX specification's own reference evaluator. Says the bytes are well formed
  #             and mean what Nx meant. Says NOTHING about whether the accelerator accepts them.
  #
  # REPLACED ORTEX, and the reason is not that ONNX Runtime was wrong. It is that the spec
  # evaluator is EXACT where a runtime is merely close -- 0 against Nx.BinaryBackend on every
  # emitted case, where ONNX Runtime 1.20.1 deviates 2.38e-07 on sqrt and 5.96e-08 on erf -- and
  # that an embedded interpreter packages into a Burrito binary where a Rust NIF plus a bundled
  # ONNX Runtime does not. Both engines accept `Erf` and `Where`, so the contrast with the DFC
  # that the secondary exists to provide is unchanged.

  alias NxShuttle.{DFC, SpecRef}

  @doc """
  Every engine that can run here, and a reason for each that cannot.

  Both are DEPLOYMENT targets, so this returns all of them rather than picking one. A defect
  that only shows up as a disagreement between them -- the cast above -- is invisible to a
  check that runs whichever happened to be available.
  """
  def engines do
    checks = [{:dfc, DFC.available()}, {:specref, SpecRef.available()}]
    {ok, missing} = Enum.split_with(checks, fn {_e, r} -> r == :ok end)
    {Enum.map(ok, &elem(&1, 0)), Enum.map(missing, fn {e, {:error, w}} -> {e, w} end)}
  end

  @doc """
  Runs `cases` through `engine`. Each case is `%{name:, model:, input:}`.

  Returns a map of name to `{:ok, tensor}` or `{:error, reason}`, in Nx's own layout.
  """
  def run(:dfc, cases), do: DFC.run(cases)

  def run(:specref, cases), do: SpecRef.run(cases)

  @doc """
  What a green run actually proved, given which engines ran.
  """
  def caveat([:dfc, :specref]),
    do: "both deployment targets ran and were cross-checked against each other"

  def caveat([:dfc]),
    do:
      "PRIMARY ONLY: the accelerator toolchain ran. Nothing was cross-checked, so a graph " <>
        "that deploys differently on the backup runtime would not show here."

  def caveat([:specref]),
    do:
      "SPEC ONLY: the ONNX reference evaluator ran. This proves the emitted graph is correct " <>
        "per the specification, NOT that the accelerator toolchain accepts it. The two the DFC " <>
        "refuses -- bare Erf, standalone Where -- both pass here, so a Hailo limitation is " <>
        "invisible in this configuration."

  def caveat([]), do: "NOTHING RAN"

  @doc """
  The highest ONNX opset each engine can load, measured rather than declared.

  The spec evaluator's ceiling is the opset its `onnx` package implements, so it moves with the
  pinned version rather than with a bundled runtime. onnx 1.17.0 implements up to 22. The
  compiler parsed 23. The emitter defaults to 17, which is under both, so nothing is currently
  blocked -- but a graph needing 23 would deploy to the accelerator and not validate against the
  specification, and that is the kind of thing that should fail a check rather than surface later.

  RETRACTED WITH ORTEX: the previous ceiling here was 21, measured on ortex 0.1.10 (21 loads, 22
  does not). That number described a bundled ONNX Runtime that is no longer in the tree.
  """
  def opset_ceiling(:specref), do: 22
  def opset_ceiling(:dfc), do: 23
end
