defmodule AxonOnnx.NxLowering do
  @moduledoc false

  # Lowers an Nx.Defn.Expr tree to ONNX nodes. Axon's IR names layers, not tensor
  # operations, so a layer whose body is an Nx function is opaque to the layer-level
  # dispatch in AxonOnnx.Serialize. Tracing the body and walking the expression gives
  # the operators that dispatch cannot reach.

  alias Onnx.NodeProto, as: Node
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.TensorProto, as: Tensor

  @doc """
  Lowers `expr` to `{output_name, nodes, initializers, state}`.

  `params` maps parameter index to the ONNX name already bound in the enclosing graph.
  `nodes` and `initializers` are returned in reverse emission order.
  """
  def lower(expr, params, state \\ %{counter: 0, cache: %{}, nodes: [], inits: []}) do
    {name, state} = do_lower(expr, params, state)
    {name, state.nodes, state.inits, state}
  end

  defp do_lower(%Nx.Tensor{data: %Nx.Defn.Expr{id: id}} = t, params, state) do
    case state.cache do
      %{^id => name} -> {name, state}
      %{} -> emit(t, params, state)
    end
  end

  defp emit(%Nx.Tensor{data: %Nx.Defn.Expr{id: id, op: op, args: args}} = t, params, state) do
    {name, state} = apply_op(op, args, t, params, state)
    {name, %{state | cache: Map.put(state.cache, id, name)}}
  end

  defp apply_op(:parameter, [index], _t, params, state) do
    {Map.fetch!(params, index), state}
  end

  defp apply_op(:constant, [value], t, _params, state) do
    {name, state} = fresh("Constant", state)
    tensor = to_tensor_proto(Nx.tensor(value, type: Nx.type(t)) |> Nx.broadcast(Nx.shape(t)))

    node = %Node{
      input: [],
      output: [name],
      name: name,
      op_type: "Constant",
      attribute: [%Attribute{name: "value", type: :TENSOR, t: tensor}]
    }

    {name, push(state, node)}
  end

  @unary %{sqrt: "Sqrt", erf: "Erf", exp: "Exp", logistic: "Sigmoid", negate: "Neg"}

  defp apply_op(op, [arg], t, params, state) when is_map_key(@unary, op) do
    {inp, state} = do_lower(arg, params, state)
    simple(Map.fetch!(@unary, op), [inp], t, state)
  end

  @binary %{add: "Add", subtract: "Sub", multiply: "Mul", divide: "Div", equal: "Equal",
            pow: "Pow", max: "Max", min: "Min"}

  defp apply_op(op, [lhs, rhs], t, params, state) when is_map_key(@binary, op) do
    {l, state} = do_lower(lhs, params, state)
    {r, state} = do_lower(rhs, params, state)
    simple(Map.fetch!(@binary, op), [l, r], t, state)
  end

  defp apply_op(:select, [pred, on_true, on_false], t, params, state) do
    {p, state} = do_lower(pred, params, state)
    {a, state} = do_lower(on_true, params, state)
    {b, state} = do_lower(on_false, params, state)
    simple("Where", [p, a, b], t, state)
  end

  defp apply_op(:as_type, [arg], t, params, state) do
    {inp, state} = do_lower(arg, params, state)
    {name, state} = fresh("Cast", state)

    node = %Node{
      input: [inp],
      output: [name],
      name: name,
      op_type: "Cast",
      attribute: [%Attribute{name: "to", type: :INT, i: onnx_type(Nx.type(t))}]
    }

    {name, push(state, node)}
  end

  defp apply_op(:reshape, [arg], t, params, state) do
    {inp, state} = do_lower(arg, params, state)
    {shape_name, state} = int64_initializer(Tuple.to_list(Nx.shape(t)), state)
    simple("Reshape", [inp, shape_name], t, state)
  end

  defp apply_op(:broadcast, [arg, shape, _axes], t, params, state) do
    _ = t
    {inp, state} = do_lower(arg, params, state)
    {shape_name, state} = int64_initializer(Tuple.to_list(shape), state)
    simple("Expand", [inp, shape_name], t, state)
  end

  defp apply_op(:transpose, [arg, axes], _t, params, state) do
    {inp, state} = do_lower(arg, params, state)
    {name, state} = fresh("Transpose", state)

    node = %Node{
      input: [inp],
      output: [name],
      name: name,
      op_type: "Transpose",
      attribute: [%Attribute{name: "perm", type: :INTS, ints: axes}]
    }

    {name, push(state, node)}
  end

  defp apply_op(:concatenate, [tensors, axis], _t, params, state) do
    {names, state} =
      Enum.map_reduce(tensors, state, fn tensor, acc -> do_lower(tensor, params, acc) end)

    {name, state} = fresh("Concat", state)

    node = %Node{
      input: names,
      output: [name],
      name: name,
      op_type: "Concat",
      attribute: [%Attribute{name: "axis", type: :INT, i: axis}]
    }

    {name, push(state, node)}
  end

  defp apply_op(:dot, [lhs, contract_l, [], rhs, contract_r, []], t, params, state) do
    rank_l = Nx.rank(lhs)

    if contract_l == [rank_l - 1] and contract_r == [0] do
      {l, state} = do_lower(lhs, params, state)
      {r, state} = do_lower(rhs, params, state)
      simple("MatMul", [l, r], t, state)
    else
      raise ArgumentError,
            "only a trailing/leading contraction lowers to MatMul, got #{inspect(contract_l)} " <>
              "against #{inspect(contract_r)}"
    end
  end

  defp apply_op(:slice, [arg, starts, lengths, strides], t, params, state) do
    {inp, state} = do_lower(arg, params, state)
    axes = Enum.to_list(0..(length(starts) - 1))
    ends = Enum.zip_with(starts, lengths, fn s, l -> s + l end)

    {starts_n, state} = int64_initializer(starts, state)
    {ends_n, state} = int64_initializer(ends, state)
    {axes_n, state} = int64_initializer(axes, state)
    {steps_n, state} = int64_initializer(strides, state)

    simple("Slice", [inp, starts_n, ends_n, axes_n, steps_n], t, state)
  end

  defp apply_op(op, _args, _t, _params, _state) do
    raise ArgumentError,
          "AxonOnnx.NxLowering has no ONNX lowering for Nx operation #{inspect(op)}"
  end

  defp simple(op_type, inputs, _t, state) do
    {name, state} = fresh(op_type, state)
    node = %Node{input: inputs, output: [name], name: name, op_type: op_type}
    {name, push(state, node)}
  end

  defp fresh(prefix, %{counter: n} = state) do
    {"nx_#{String.downcase(prefix)}_#{n}", %{state | counter: n + 1}}
  end

  defp push(state, node), do: %{state | nodes: [node | state.nodes]}

  defp int64_initializer(values, state) do
    {name, state} = fresh("Init", state)

    tensor = %Tensor{
      name: name,
      dims: [length(values)],
      data_type: 7,
      raw_data: for(v <- values, into: <<>>, do: <<v::signed-little-64>>)
    }

    {name, %{state | inits: [tensor | state.inits]}}
  end

  defp to_tensor_proto(tensor) do
    %Tensor{
      dims: Tuple.to_list(Nx.shape(tensor)),
      data_type: onnx_type(Nx.type(tensor)),
      raw_data: Nx.to_binary(tensor)
    }
  end

  @doc false
  def onnx_type({:f, 32}), do: 1
  def onnx_type({:u, 8}), do: 2
  def onnx_type({:s, 8}), do: 3
  def onnx_type({:s, 16}), do: 5
  def onnx_type({:s, 32}), do: 6
  def onnx_type({:s, 64}), do: 7
  def onnx_type({:f, 64}), do: 11
  def onnx_type({:u, 1}), do: 9
  def onnx_type(:u8), do: 2

  def onnx_type(type),
    do: raise(ArgumentError, "no ONNX data type for #{inspect(type)}")
end
