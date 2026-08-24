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
    with {_, 0} <-
           System.cmd("docker", ["version", "--format", "{{.Server.Os}}"], stderr_to_stdout: true),
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

  Returns a map of name to `{:ok, %{native: tensor, bit_exact: tensor}}` or `{:error, reason}`.

  TWO CONTEXTS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS. `SDK_NATIVE` runs the parsed graph in
  float and says whether the lowering is structurally right. `SDK_BIT_EXACT` runs the QUANTIZED
  graph, which is what actually reaches the accelerator, and says what deployment will compute.
  Measured on `(x + 1) * 0.5`, native agrees with Nx to 0.0 and bit-exact diverges by 0.455.
  Reporting only the first would certify a graph nobody deploys.
  """
  def run(cases), do: run(cases, [])

  @doc """
  As `run/1`, with `:script` -- a Hailo model script applied before `optimize/1`.

  `precision_script/1` builds the rungs. Nothing is applied by default, which keeps `run/1`
  reporting what the compiler does when nobody has intervened.
  """
  def run(cases, opts) do
    dir = Path.join(System.tmp_dir!(), "nx_shuttle_dfc_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    try do
      manifest =
        Enum.map(cases, fn %{name: name, model: model, input: input} ->
          File.write!(Path.join(dir, "#{name}.onnx"), model)
          nhwc = Nx.transpose(input, axes: [0, 2, 3, 1])
          File.write!(Path.join(dir, "#{name}.in"), Nx.to_binary(nhwc))

          # CALIBRATE OVER THE RANGE THE INPUT ACTUALLY SPANS. A first version calibrated on a
          # fixed 0..2 while the inputs reached 4.375, so everything above 2 saturated and the
          # quantization error it reported -- up to 2.375 on a plain Add -- was measuring the
          # harness rather than the target. Quantization is only as good as the range it was
          # shown, and the range has to come from the data.
          %{
            "name" => name,
            "in_shape" => Tuple.to_list(Nx.shape(nhwc)),
            "lo" => input |> Nx.reduce_min() |> Nx.to_number(),
            "hi" => input |> Nx.reduce_max() |> Nx.to_number(),
            "script" => Keyword.get(opts, :script)
          }
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
            read = fn ext ->
              Path.join(dir, "#{name}.#{ext}")
              |> File.read!()
              |> Nx.from_binary({:f, 32})
              |> Nx.reshape(List.to_tuple(shape))
              |> then(fn t ->
                if Nx.rank(t) == 4, do: Nx.transpose(t, axes: [0, 3, 1, 2]), else: t
              end)
            end

            _ = input
            {name, {:ok, %{native: read.("native"), bit_exact: read.("exact")}}}

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

  @doc """
  A model script for one rung of the precision ladder, or `nil` for the compiler's own default.

  THE RUNGS THAT EXIST, which is not what the release notes advertise. 5.3.0 announces `a16_w8`
  and `a16_w4` as Hailo-10H preview and the compiler rejects both outright with
  `Unsupported value [PrecisionMode.a16_w8] for fields ['precision_mode']`. What is left is one
  ratio dial and the default.

      :default   whatever optimize() does unaided -- a8_w8
      :w4        auto_4bit_weights_ratio=1
      :a16_w16   auto_16bit_weights_ratio=1

  THE DIAL IS NOT WEIGHTS-ONLY, despite its name, and the compiler said so rather than the
  documentation. Rejecting `pow` under this script it reported the config it had built:
  `Layer: pow/square1, Config: {'precision_mode': 'a16_w16', ...}`. So
  `auto_16bit_weights_ratio=1` yields a16_w16, activations included. An earlier version of this
  function called that rung `:w16` and offered a separate `:a16_w16` that added
  `optimization_level=2`; that was two names for one precision and a second dial wearing the
  first one's label. Measured, they differed by 0.000013 on `add` -- the optimization level, not
  the bit width.

  MEASURED, on `x` over [0.5, 5.375], max |bit_exact - Nx|:

      case      :w4                     :default    :a16_w16
      add       ZeroDivisionError       0.019428    0.000166
      sigmoid   ZeroDivisionError       0.007119    0.006048
      exp       ZeroDivisionError       0.483141    0.333931
      pow       ZeroDivisionError       0.078688    AccelerasImplementationError

  Three things that table says and a summary would not. `:w4` is not a rung: it fails on every
  graph here, including `add`, with `ZeroDivisionError: division by zero`. `pow` gets WORSE by
  going up -- 16-bit is unsupported for its square layer, so its only workable rung is the
  default. And `exp` is not a precision problem at all: 0.33 at a16_w16 is still unusable, and
  the cause is dynamic range -- e^0.5 to e^5.375 spans 1.6 to 216 -- which more bits on the
  weights do not address.

  So there is no single lowest rung. It is per-graph, and `add` improving 117x while `pow`
  stops compiling is the reason this returns a script rather than picking one.
  """
  def precision_script(:default), do: nil

  def precision_script(:w4),
    do: "model_optimization_config(compression_params, auto_4bit_weights_ratio=1)"

  def precision_script(:a16_w16),
    do: "model_optimization_config(compression_params, auto_16bit_weights_ratio=1)"

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
                native = np.asarray(runner.infer(ctx, x)).astype(np.float32)

            # 1024 is the amount the optimizer asks for; below it, it drops to optimization
            # level 0 and says so, and a quantization figure measured there is not the one
            # deployment would see.
            rng = np.random.RandomState(0)
            lo, hi = float(case["lo"]), float(case["hi"])
            pad = 0.05 * max(abs(lo), abs(hi), 1.0)
            calib = rng.uniform(lo - pad, hi + pad, size=[1024] + list(shape[1:])).astype(np.float32)
            # PRECISION IS A DIAL, and until now it was never turned. optimize() at defaults is
            # a8_w8. A graph whose arithmetic does not survive that is not necessarily one that
            # cannot deploy -- it is one that needs more bits. The caller names the rung and the
            # point is to report the LOWEST that works rather than the safest.
            script = case.get("script")
            if script:
                runner.load_model_script(script)
            runner.optimize(calib)

            with runner.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
                exact = np.asarray(runner.infer(ctx, x)).astype(np.float32)

            open(f"/work/{name}.native", "wb").write(native.tobytes())
            open(f"/work/{name}.exact", "wb").write(exact.tobytes())
            results[name] = {"ok": True, "shape": list(native.shape)}
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    json.dump(results, open("/work/results.json", "w"))
    """
  end
end
