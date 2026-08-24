defmodule AxonOnnx.NxLoweringTest do
  use ExUnit.Case, async: false

  alias AxonOnnx.NxLowering

  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  # DFC 5.3.0 takes opset 15-21. 17 is the floor that also has LayerNormalization.
  @opset 17

  defp value_info(name, tensor) do
    dims = Nx.shape(tensor) |> Tuple.to_list() |> Enum.map(&%Dimension{value: {:dim_value, &1}})

    %Value{
      name: name,
      type: %Type{
        value:
          {:tensor_type,
           %Placeholder{
             elem_type: NxLowering.onnx_type(Nx.type(tensor)),
             shape: %Shape{dim: dims}
           }}
      }
    }
  end

  defp build(fun, args) do
    templates = Enum.map(args, &Nx.template(Nx.shape(&1), Nx.type(&1)))
    expr = apply(Nx.Defn.debug_expr(fun), templates)

    params = for {_a, i} <- Enum.with_index(args), into: %{}, do: {i, "input_#{i}"}
    {out, nodes, inits, _state} = NxLowering.lower(expr, params)

    graph = %Graph{
      node: Enum.reverse(nodes),
      name: "test",
      input: for({a, i} <- Enum.with_index(args), do: value_info("input_#{i}", a)),
      output: [value_info(out, expr)],
      initializer: Enum.reverse(inits)
    }

    %Model{
      ir_version: 8,
      producer_name: "AxonOnnxTest",
      opset_import: [%Opset{domain: "", version: @opset}],
      graph: graph
    }
  end

  setup_all do
    # Torchx is the backend on this platform: XLA publishes no windows-x86_64 archive, so
    # the expected values come from libtorch rather than from the pure-Elixir fallback.
    Nx.default_backend(Torchx.Backend)

    # An unmet precondition is a failure, not a skip: a suite that quietly runs zero
    # comparisons reports the same green as one that ran them all.
    case System.cmd(python(), ["-c", "import onnx, onnxruntime"], stderr_to_stdout: true) do
      {_, 0} ->
        :ok

      {out, code} ->
        raise "#{python()} cannot import onnx/onnxruntime (exit #{code}): #{String.trim(out)}"
    end
  end

  defp python, do: System.get_env("ONNX_PYTHON") || "python3"

  defp write_tensor(tensor, path) do
    tp = %Onnx.TensorProto{
      dims: Nx.shape(tensor) |> Tuple.to_list(),
      data_type: NxLowering.onnx_type(Nx.type(tensor)),
      raw_data: Nx.to_binary(tensor)
    }

    File.write!(path, Onnx.TensorProto.encode!(tp))
    path
  end

  defp read_tensor(path) do
    %Onnx.TensorProto{dims: dims, data_type: 1, raw_data: raw} =
      path |> File.read!() |> Onnx.TensorProto.decode!()

    raw |> Nx.from_binary({:f, 32}) |> Nx.reshape(List.to_tuple(dims))
  end

  defp run_in_ort(model, args) do
    dir = Path.join(System.tmp_dir!(), "lowering_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    model_path = Path.join(dir, "model.onnx")
    out_path = Path.join(dir, "out.pb")
    File.write!(model_path, Model.encode!(model))

    in_paths =
      args
      |> Enum.with_index()
      |> Enum.map(fn {a, i} -> write_tensor(a, Path.join(dir, "in_#{i}.pb")) end)

    try do
      case System.cmd(python(), ["scripts/run_onnx_case.py", model_path, out_path | in_paths],
             stderr_to_stdout: true
           ) do
        {_, 0} -> read_tensor(out_path)
        {out, code} -> raise "onnxruntime run failed (exit #{code}): #{String.trim(out)}"
      end
    after
      File.rm_rf(dir)
    end
  end

  defp assert_agrees(fun, args, opts \\ []) do
    tol = opts[:tol] || 1.0e-5
    expected = apply(fun, args)
    got = run_in_ort(build(fun, args), args)

    diff =
      Nx.subtract(Nx.as_type(got, {:f, 32}), Nx.as_type(expected, {:f, 32}))
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff <= tol, "ONNX Runtime and Nx disagree by #{diff}, tolerance #{tol}"
    diff
  end

  @a Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], type: :f32)
  @b Nx.tensor([[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]], type: :f32)

  describe "the operators the hailo10h device half needs" do
    test "Add" do
      assert_agrees(&Nx.add/2, [@a, @b])
    end

    test "Mul" do
      assert_agrees(&Nx.multiply/2, [@a, @b])
    end

    test "Div" do
      assert_agrees(&Nx.divide/2, [@a, @b])
    end

    test "Sqrt" do
      assert_agrees(fn x -> Nx.sqrt(x) end, [@a])
    end

    test "Erf" do
      assert_agrees(fn x -> Nx.erf(x) end, [@a])
    end

    test "Equal and Where" do
      assert_agrees(
        fn x, y -> Nx.select(Nx.equal(x, y), x, Nx.multiply(y, 2.0)) end,
        [@a, @b]
      )
    end

    test "Cast" do
      assert_agrees(fn x -> Nx.as_type(x, {:s, 32}) |> Nx.as_type({:f, 32}) end, [@a])
    end

    test "Reshape" do
      assert_agrees(fn x -> Nx.reshape(x, {4, 2}) end, [@a])
    end

    test "Transpose" do
      assert_agrees(fn x -> Nx.transpose(x) end, [@a])
    end

    test "MatMul" do
      assert_agrees(fn x, y -> Nx.dot(x, Nx.transpose(y)) end, [@a, @b])
    end

    test "Concat" do
      assert_agrees(fn x, y -> Nx.concatenate([x, y], axis: 0) end, [@a, @b])
    end

    test "Expand" do
      assert_agrees(fn x -> Nx.broadcast(x, {3, 2, 4}) end, [@a])
    end

    test "Slice" do
      assert_agrees(fn x -> Nx.slice(x, [0, 1], [2, 2]) end, [@a])
    end

    test "a composite chain, which is where operand order goes wrong" do
      fun = fn x, y ->
        n = Nx.divide(Nx.subtract(x, y), Nx.sqrt(Nx.add(Nx.multiply(y, y), 1.0)))
        Nx.dot(Nx.transpose(n), Nx.select(Nx.equal(x, y), n, Nx.erf(n)))
      end

      assert_agrees(fun, [@a, @b])
    end
  end

  describe "negative controls" do
    test "a subtraction lowered with its operands swapped must NOT agree" do
      # Sub is not commutative, so a swapped lowering is exactly the class of bug the
      # agreement test above exists to catch. If this passes, the test is decoration.
      fun = fn x, y -> Nx.subtract(x, y) end
      swapped = fn x, y -> Nx.subtract(y, x) end

      expected = fun.(@a, @b)
      got = run_in_ort(build(swapped, [@a, @b]), [@a, @b])

      diff =
        Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff > 1.0e-3,
             "a swapped subtraction agreed to #{diff}; the comparison cannot see operand order"
    end

    test "an Nx operation with no lowering raises rather than emitting a silent nothing" do
      assert_raise ArgumentError, ~r/no ONNX lowering for Nx operation/, fn ->
        build(fn x -> Nx.sort(x, axis: 0) end, [@a])
      end
    end

    test "a dot that is not a trailing/leading contraction raises" do
      assert_raise ArgumentError, ~r/only a trailing\/leading contraction/, fn ->
        build(fn x, y -> Nx.dot(x, [0], y, [0]) end, [@a, @b])
      end
    end
  end
end
