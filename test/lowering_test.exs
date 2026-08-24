defmodule NxOnnx.LoweringTest do
  use ExUnit.Case, async: false

  # The reference is ONNX Runtime, in-process through Pythonx. Comparing the lowering against
  # anything in this repository would only prove it agrees with itself; the question is whether
  # a runtime that did not know how the graph was built reads it the way Nx meant it.

  setup_all do
    Nx.default_backend(Torchx.Backend)

    # An unmet precondition is a failure, not a skip: a suite that quietly runs zero
    # comparisons reports the same green as one that ran them all.
    Pythonx.eval(
      """
      import numpy, onnx, onnxruntime
      from onnx import numpy_helper
      """,
      %{}
    )

    :ok
  end

  defp run_in_ort(model, args) do
    bytes = NxOnnx.encode(model)

    inputs =
      model.graph.input
      |> Enum.zip(args)
      |> Map.new(fn {%{name: name}, t} ->
        {name, {Nx.to_binary(t), Tuple.to_list(Nx.shape(t))}}
      end)

    {result, _globals} =
      Pythonx.eval(
        """
        import numpy, onnx, onnxruntime
        model = onnx.load_model_from_string(bytes(model_bytes))
        onnx.checker.check_model(model)
        def name(k):
            return k.decode() if isinstance(k, bytes) else k
        feed = {name(k): numpy.frombuffer(bytes(v[0]), dtype=numpy.float32).reshape([int(d) for d in v[1]])
                for k, v in inputs.items()}
        sess = onnxruntime.InferenceSession(bytes(model_bytes), providers=["CPUExecutionProvider"])
        out = sess.run(None, feed)[0].astype(numpy.float32)
        (out.tobytes(), list(out.shape))
        """,
        %{"model_bytes" => bytes, "inputs" => inputs}
      )

    {raw, shape} = Pythonx.decode(result)
    raw |> Nx.from_binary({:f, 32}) |> Nx.reshape(List.to_tuple(shape))
  end

  defp assert_agrees(fun, args, opts \\ []) do
    tol = opts[:tol] || 1.0e-5
    expected = apply(fun, args)
    templates = Enum.map(args, &Nx.template(Nx.shape(&1), Nx.type(&1)))
    got = run_in_ort(NxOnnx.to_model!(fun, templates), args)

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
    test "Add", do: assert_agrees(&Nx.add/2, [@a, @b])
    test "Mul", do: assert_agrees(&Nx.multiply/2, [@a, @b])
    test "Div", do: assert_agrees(&Nx.divide/2, [@a, @b])
    test "Sqrt", do: assert_agrees(fn x -> Nx.sqrt(x) end, [@a])
    test "Erf", do: assert_agrees(fn x -> Nx.erf(x) end, [@a])
    test "Reshape", do: assert_agrees(fn x -> Nx.reshape(x, {4, 2}) end, [@a])
    test "Transpose", do: assert_agrees(fn x -> Nx.transpose(x) end, [@a])
    test "Expand", do: assert_agrees(fn x -> Nx.broadcast(x, {3, 2, 4}) end, [@a])
    test "Slice", do: assert_agrees(fn x -> Nx.slice(x, [0, 1], [2, 2]) end, [@a])
    test "MatMul", do: assert_agrees(fn x, y -> Nx.dot(x, Nx.transpose(y)) end, [@a, @b])
    test "Concat", do: assert_agrees(fn x, y -> Nx.concatenate([x, y], axis: 0) end, [@a, @b])

    test "Equal and Where" do
      assert_agrees(fn x, y -> Nx.select(Nx.equal(x, y), x, Nx.multiply(y, 2.0)) end, [@a, @b])
    end

    test "Cast" do
      assert_agrees(fn x -> Nx.as_type(x, {:s, 32}) |> Nx.as_type({:f, 32}) end, [@a])
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
      # agreement tests exist to catch. If this passes, the comparison is decoration.
      templates = [Nx.template({2, 4}, :f32), Nx.template({2, 4}, :f32)]
      expected = Nx.subtract(@a, @b)
      got = run_in_ort(NxOnnx.to_model!(fn x, y -> Nx.subtract(y, x) end, templates), [@a, @b])

      diff = Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff > 1.0e-3,
             "a swapped subtraction agreed to #{diff}; the comparison cannot see operand order"
    end

    test "an Nx operation with no lowering is reported, not silently dropped" do
      templates = [Nx.template({2, 4}, :f32)]

      assert {:error, message} = NxOnnx.to_model(fn x -> Nx.sort(x, axis: 0) end, templates)
      assert message =~ "no ONNX lowering for Nx operation"
    end

    test "a dot that is not a trailing/leading contraction is reported" do
      templates = [Nx.template({2, 4}, :f32), Nx.template({2, 4}, :f32)]

      assert {:error, message} =
               NxOnnx.to_model(fn x, y -> Nx.dot(x, [0], y, [0]) end, templates)

      assert message =~ "only a trailing/leading contraction"
    end
  end
end
