defmodule NxShuttle.DFC do
  @moduledoc false

  # Runs emitted graphs through the Hailo Dataflow Compiler and returns what it computed.
  #
  # The DFC is the consumer this library exists to satisfy, so it is the reference. A
  # general-purpose runtime agreeing with Nx says nothing about whether the graph reaches the
  # accelerator; only the compiler that has to accept it can say that.
  #
  # It ships as a licensed wheel and runs in a container, so it cannot be embedded. Cases are
  # therefore batched into one container invocation rather than one per case.

  @default_image "weftspun-hailo-dfc:latest"
  @arch "hailo10h"

  # Read at runtime rather than baked in, so the fallback path can be exercised by pointing
  # this at an image that does not exist. A selection rule nothing ever takes is a rule nobody
  # has checked.
  def image, do: System.get_env("NX_SHUTTLE_DFC_IMAGE") || @default_image

  @doc """
  Returns `:ok` when the compiler can actually be run, `{:error, reason}` otherwise.
  """
  def available do
    with {_, 0} <- System.cmd("docker", ["version", "--format", "{{.Server.Os}}"], stderr_to_stdout: true),
         {out, 0} <-
           System.cmd("docker", ["images", "-q", image()], stderr_to_stdout: true) do
      if String.trim(out) == "", do: {:error, "docker image #{image()} is not built"}, else: :ok
    else
      {out, code} -> {:error, "docker unavailable (exit #{code}): #{String.trim(out)}"}
    end
  end

  @doc """
  Compiles and runs `cases` through the DFC.

  Each case is `%{name:, model: binary, input: Nx.Tensor.t()}`. The ONNX input is NCHW by
  convention and the DFC works in NHWC, so inputs are transposed on the way in and outputs on
  the way back, leaving both sides comparable in Nx's own layout.

  Returns a map of name to `{:ok, tensor}` or `{:error, reason}`.
  """
  def run(cases) do
    dir = Path.join(System.tmp_dir!(), "nx_shuttle_dfc_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    try do
      manifest =
        Enum.map(cases, fn %{name: name, model: model, input: input} ->
          File.write!(Path.join(dir, "#{name}.onnx"), model)
          nhwc = Nx.transpose(input, axes: [0, 2, 3, 1])
          File.write!(Path.join(dir, "#{name}.in"), Nx.to_binary(nhwc))

          %{"name" => name, "in_shape" => Tuple.to_list(Nx.shape(nhwc))}
        end)

      File.write!(Path.join(dir, "manifest.json"), Jason.encode!(manifest))
      File.write!(Path.join(dir, "run.py"), runner_py())

      {out, code} =
        System.cmd(
          "docker",
          ["run", "--rm", "-v", "#{windows_path(dir)}:/work", image(), "python", "/work/run.py"],
          stderr_to_stdout: true,
          env: [{"MSYS_NO_PATHCONV", "1"}]
        )

      if code != 0 do
        raise "DFC container exited #{code}:\n#{String.slice(out, -3000, 3000)}"
      end

      results = dir |> Path.join("results.json") |> File.read!() |> Jason.decode!()

      Map.new(cases, fn %{name: name, input: input} ->
        case results[name] do
          %{"ok" => true, "shape" => shape} ->
            raw = File.read!(Path.join(dir, "#{name}.out"))

            got =
              raw
              |> Nx.from_binary({:f, 32})
              |> Nx.reshape(List.to_tuple(shape))
              |> then(fn t ->
                if Nx.rank(t) == 4, do: Nx.transpose(t, axes: [0, 3, 1, 2]), else: t
              end)

            _ = input
            {name, {:ok, got}}

          %{"ok" => false, "error" => why} ->
            {name, {:error, why}}

          nil ->
            {name, {:error, "the container reported nothing for this case"}}
        end
      end)
    after
      File.rm_rf(dir)
    end
  end

  # Windows paths reach docker as-is; POSIX ones are already correct.
  defp windows_path(path) do
    case :os.type() do
      {:win32, _} -> String.replace(path, "\\", "/")
      _ -> path
    end
  end

  defp runner_py do
    """
    import json, warnings, numpy as np
    warnings.filterwarnings("ignore")
    from hailo_sdk_client import ClientRunner, InferenceContext

    manifest = json.load(open("/work/manifest.json"))
    results = {}

    for case in manifest:
        name = case["name"]
        try:
            runner = ClientRunner(hw_arch="#{@arch}")
            runner.translate_onnx_model(f"/work/{name}.onnx", name.replace("-", "_"))
            shape = [int(d) for d in case["in_shape"]]
            x = np.frombuffer(open(f"/work/{name}.in", "rb").read(), dtype=np.float32).reshape(shape)
            with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
                out = np.asarray(runner.infer(ctx, x)).astype(np.float32)
            open(f"/work/{name}.out", "wb").write(out.tobytes())
            results[name] = {"ok": True, "shape": list(out.shape)}
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    json.dump(results, open("/work/results.json", "w"))
    """
  end
end
