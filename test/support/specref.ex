defmodule NxShuttle.SpecRef do
  @moduledoc false

  # Runs emitted graphs through the ONNX specification's own reference evaluator.
  #
  # WHY THE SPEC EVALUATOR RATHER THAN A RUNTIME. `onnx.reference.ReferenceEvaluator` is plain
  # numpy per the specification: no fused kernels, no fast-math, no vendor opinion about what an
  # operator ought to do. Measured against Nx.BinaryBackend on the emitted graphs, it agrees
  # BIT-EXACTLY, where ONNX Runtime 1.20.1 does not:
  #
  #     case    onnxruntime   spec evaluator
  #     add     0             0
  #     sqrt    2.38e-07      0
  #     erf     5.96e-08      0
  #     where   0             0
  #
  # Exactness is what a verifier is for. A runtime tells you what one vendor's kernels compute;
  # the reference tells you what the graph MEANS.
  #
  # WHAT IT IS FOR, which is not "a second opinion on arithmetic". Its job is to be a PERMISSIVE
  # standard consumer next to a restrictive one. The DFC refuses `Erf` standing alone and `Where`
  # with nothing to fold it into; the spec evaluator runs both and agrees. That contrast is what
  # separates "our lowering is wrong" from "the accelerator will not take this", and it is how
  # the Erf/Gelu and Where/fold findings in README.md were found in the first place.
  #
  # It is Python, embedded through pythonx rather than shelled out to, because `weft-warp-burrito`
  # packages this into one executable and an executable cannot call an environment that is not
  # inside it. The dependency set is pinned in `@pyproject` below, which is also what exempts it
  # from the `uv` blocklist entry -- see BLOCKLIST.md, "an embedded interpreter that declares its
  # pins in source is exempt".

  @pyproject """
  [project]
  name = "nx_shuttle_specref"
  version = "0.0.0"
  requires-python = "==3.11.*"
  dependencies = ["onnx==1.17.0", "numpy"]
  """

  @probe_key {__MODULE__, :available}

  @doc """
  Returns `:ok` when the evaluator can actually be run, `{:error, reason}` otherwise.

  PROVES IT RATHER THAN ASSUMING IT. The ortex probe this replaces asked `Code.ensure_loaded?`,
  which was honest for a NIF: the module is there or it is not. An embedded interpreter cannot be
  probed that way. `Pythonx` is always loaded, and the thing that fails is `uv_init` resolving a
  Python and two packages -- which needs a network the first time and can fail for reasons the
  module list knows nothing about. So this runs the import once and keeps the verdict.

  Caching is not an optimisation here. `uv_init/1` raises if called twice in a VM, so the result
  has to be remembered rather than recomputed, and twelve cases must not each try to initialise.
  """
  def available do
    case :persistent_term.get(@probe_key, :unprobed) do
      :unprobed ->
        result = probe()
        :persistent_term.put(@probe_key, result)
        result

      cached ->
        cached
    end
  end

  defp probe do
    Pythonx.uv_init(@pyproject)

    {_, _} =
      Pythonx.eval("import onnx, numpy\nfrom onnx.reference import ReferenceEvaluator\n1", %{})

    :ok
  rescue
    e -> {:error, "onnx reference evaluator unavailable: " <> Exception.message(e)}
  end

  @doc """
  Runs `cases` through the reference evaluator.

  Each case is `%{name:, model: binary, input: Nx.Tensor.t()}`. Returns a map of name to
  `{:ok, tensor}` or `{:error, reason}`, in Nx's own layout -- the same shape `DFC.run/1` returns,
  so `Reference.run/2` can dispatch on the engine and nothing downstream cares which ran.

  A REJECTION IS A RESULT, NOT A CRASH. The DFC refuses two of the twelve cases outright, and the
  point of this engine is to say what a compliant consumer does with the same bytes. So a graph
  the evaluator will not load returns `{:error, reason}` for that case and the others still run.
  """
  def run(cases) do
    dir = Path.join(System.tmp_dir!(), "nx_shuttle_specref_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    try do
      names =
        Enum.map(cases, fn %{name: name, model: model, input: input} ->
          File.write!(Path.join(dir, "#{name}.onnx"), model)
          File.write!(Path.join(dir, "#{name}.in"), Nx.to_binary(input))
          {name, Nx.shape(input)}
        end)

      {report, _} =
        Pythonx.eval(python_source(), %{
          "dir" => Pythonx.encode!(dir),
          "names" => Pythonx.encode!(Enum.map(names, &elem(&1, 0))),
          "shape" => Pythonx.encode!(names |> hd() |> elem(1) |> Tuple.to_list())
        })

      report
      |> Pythonx.decode()
      |> String.split("\n", trim: true)
      |> Map.new(&decode_row(&1, dir))
    after
      File.rm_rf(dir)
    end
  end

  # One line per case: `name<TAB>OK<TAB>d1,d2,...` or `name<TAB>ERR<TAB>message`. The output
  # tensor comes back through a file rather than through the bridge, because the bridge would
  # have to agree with Nx about dtype and layout and a raw float32 blob does not.
  defp decode_row(line, dir) do
    case String.split(line, "\t") do
      [name, "OK", dims] ->
        shape =
          dims
          |> String.split(",", trim: true)
          |> Enum.map(&String.to_integer/1)
          |> List.to_tuple()

        tensor =
          Path.join(dir, "#{name}.out")
          |> File.read!()
          |> Nx.from_binary(:f32)
          |> Nx.reshape(shape)

        {name, {:ok, tensor}}

      [name, "ERR", why] ->
        {name, {:error, why}}
    end
  end

  defp python_source do
    ~S'''
    import numpy as np, os
    from onnx import load
    from onnx.reference import ReferenceEvaluator

    d = dir.decode() if isinstance(dir, bytes) else str(dir)
    shp = tuple(int(v) for v in shape)
    rows = []
    for raw in names:
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        try:
            x = np.fromfile(os.path.join(d, name + ".in"), dtype=np.float32).reshape(shp)
            ev = ReferenceEvaluator(load(os.path.join(d, name + ".onnx")))
            out = np.asarray(ev.run(None, {ev.input_names[0]: x})[0], dtype=np.float32)
            out.tofile(os.path.join(d, name + ".out"))
            rows.append("%s\tOK\t%s" % (name, ",".join(str(v) for v in out.shape)))
        except Exception as e:
            rows.append("%s\tERR\t%s" % (name, str(e).replace("\t", " ").replace("\n", " ")[:200]))
    "\n".join(rows)
    '''
  end
end
