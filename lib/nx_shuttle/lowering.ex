defmodule NxShuttle.Lowering do
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

  # `sigmoid`, not `logistic`. The key here read `logistic:` and could never fire: Nx has no
  # `:logistic` op -- `Nx.sigmoid/1` builds `:sigmoid` -- so `Sigmoid` was unreachable and
  # `Nx.sigmoid` raised "no ONNX lowering" while a Sigmoid mapping sat right here. Found by
  # sweeping every operator this module claims to emit rather than the seven the suite covered.
  @unary %{
    sqrt: "Sqrt",
    erf: "Erf",
    exp: "Exp",
    sigmoid: "Sigmoid",
    negate: "Neg",
    abs: "Abs",
    ceil: "Ceil",
    cos: "Cos",
    log: "Log",
    sign: "Sign",
    sin: "Sin",
    tanh: "Tanh"
  }

  defp apply_op(op, [arg], t, params, state) when is_map_key(@unary, op) do
    {inp, state} = do_lower(arg, params, state)
    simple(Map.fetch!(@unary, op), [inp], t, state)
  end

  @binary %{
    add: "Add",
    subtract: "Sub",
    multiply: "Mul",
    divide: "Div",
    equal: "Equal",
    greater: "Greater",
    greater_equal: "GreaterOrEqual",
    less: "Less",
    less_equal: "LessOrEqual",
    pow: "Pow",
    max: "Max",
    min: "Min"
  }

  defp apply_op(op, [lhs, rhs], t, params, state) when is_map_key(@binary, op) do
    {l, state} = do_lower(lhs, params, state)
    {r, state} = do_lower(rhs, params, state)
    simple(Map.fetch!(@binary, op), [l, r], t, state)
  end

  # FLOOR IS REFUSED, and not because it is unsupported. The accelerator ACCEPTS it and does not
  # perform it, which is the same defect the Cast rule below names and the reason that rule
  # exists. Measured on x over [0.5, 5.375] step 0.125, against the compiler's own emulator:
  #
  #     max |dfc - floor(x)|   0.875     (the largest fractional part in the input)
  #     max |dfc - x|          0.0       (it returned the input, untouched)
  #
  #     in    [0.5, 0.625, 0.75, 0.875]
  #     out   [0.5, 0.625, 0.75, 0.875]
  #     floor [0.0, 0.0,   0.0,  0.0  ]
  #
  # The spec evaluator computes it exactly, so the lowering was right and the target is wrong.
  # A silently wrong number is worse than a refusal, so this refuses.
  defp apply_op(:floor, [_arg], _t, _params, _state) do
    raise ArgumentError,
          "refusing to emit Floor: the target accepts it and returns its input unchanged " <>
            "(measured max|dfc - x| = 0.0 where floor would have moved every element), so the " <>
            "graph would compute something other than the function it came from"
  end

  # ROUND IS REFUSED FOR THE OPPOSITE REASON: the operator is fine and the SEMANTICS differ.
  # `Nx.round/1` rounds half away from zero and ONNX `Round` rounds half to even. Measured on
  # [0.5, 1.5, 2.5, -0.5]:
  #
  #     Nx     [1.0, 2.0, 3.0, -1.0]
  #     ONNX   [0.0, 2.0, 2.0, -0.0]
  #
  # This was caught by the spec evaluator disagreeing by 1.0, which is what a permissive second
  # engine is carried for -- the accelerator merely refused the node and would never have shown
  # which of the two roundings the graph had asked for.
  #
  # It is refused rather than approximated because the specification has no half-away-from-zero
  # rounding, and `sign(x) * floor(abs(x) + 0.5)` needs Floor, which is refused above.
  defp apply_op(:round, [_arg], _t, _params, _state) do
    raise ArgumentError,
          "refusing to emit Round: Nx rounds half away from zero and ONNX Round rounds half " <>
            "to even, so the emitted graph would disagree with the function it came from at " <>
            "every exact half (measured 1.0 on 0.5)"
  end

  # LOGICAL_AND AS A MASK AND A MULTIPLY, which is the same shape the Where decomposition uses
  # and the only one this target accepts. ONNX `And` takes booleans and Nx applies logical_and
  # to floats, so the obvious route is Cast-to-bool then And -- refused here by the Cast rule
  # below, because a float-to-bool Cast changes values and this target accepts a value-changing
  # Cast without performing it.
  #
  # The mask is `min(max(|x| * k, eps), 1)`: 1 wherever x is nonzero, 0 at zero, built from Abs,
  # Mul, Max and Min. Conjunction is the PRODUCT of the two masks.
  #
  # MEASURED, and the three forms do not fare alike:
  #
  #     min(abs(sign a), abs(sign b))   REJECTED TypeError
  #     min(mask a, mask b)             REJECTED AccelerasValueError
  #     mul(mask a, mask b)             compiles, float diff 0.0
  #
  # So `Mul` rather than `Min` even though both are conjunction on zero-or-one values, and no
  # `Sign` -- Sign standing alone is refused by this target (InvalidHNError). An earlier version
  # of this clause used abs(sign(x)) and could not run on the accelerator at all.
  #
  # `eps` is -1.0e-6 rather than 0.0 for the reason recorded in the logbook: an exact-zero floor
  # is recognised as ReLU, which carries no clip_min, and the paired Min then fails with
  # KeyError: 'clip_min'.
  #
  # THE OUTPUT TYPE DIFFERS AND IT IS DELIBERATE. Nx types logical_and as {:u, 8}; this emits
  # float 0.0 or 1.0. The values agree and the labels do not. A Cast to fix the label would
  # reintroduce exactly what the first paragraph refuses.
  defp apply_op(:logical_and, [lhs, rhs], t, params, state) do
    {l, state} = do_lower(lhs, params, state)
    {r, state} = do_lower(rhs, params, state)

    # THE MASK IS BUILT IN THE OPERAND'S TYPE, NOT THE RESULT'S. Nx types logical_and as
    # {:u, 8}, and the scale constant 1.0e6 does not fit in a u8 -- building the constants
    # against the output type fails at proto construction, which is how this was found.
    ft = Nx.template(Nx.shape(t), Nx.type(lhs))

    {lm, state} = nonzero_mask(l, ft, state)
    {rm, state} = nonzero_mask(r, ft, state)
    simple("Mul", [lm, rm], t, state)
  end

  # MOD DEFAULTS TO INTEGER MODULO, and float inputs need `fmod=1` or the compiler is being
  # asked for the wrong function. Nx.remainder follows C fmod -- the sign of the dividend --
  # which is what fmod=1 selects.
  defp apply_op(:remainder, [lhs, rhs], _t, params, state) do
    {l, state} = do_lower(lhs, params, state)
    {r, state} = do_lower(rhs, params, state)
    {name, state} = fresh("Mod", state)

    node = %Node{
      input: [l, r],
      output: [name],
      name: name,
      op_type: "Mod",
      attribute: [%Attribute{name: "fmod", type: :INT, i: 1}]
    }

    {name, push(state, node)}
  end

  # LOGICAL_NOT ARRIVES AS A BLOCK, not as an operator, and the struct is matched rather than
  # the tag. `Nx.logical_not/1` builds `:block` carrying `%Nx.Block.LogicalNot{}` with a
  # fallback that expands to `equal(t, 0)` -- and `Equal` is refused by the accelerator, so
  # taking the fallback would lower a working operator into a rejected one. ONNX `Not` is the
  # direct emission.
  #
  # THE MATCH IS NARROW ON PURPOSE. `Nx.Block` has 21 kinds -- Phase, TopK, CumulativeSum,
  # FFT2, the whole LinAlg family. A clause that matched `:block` alone would silently accept
  # every one of them and emit `Not`, which is the failure this module's catch-all exists to
  # prevent. Everything that is not LogicalNot falls through to "no ONNX lowering", and a
  # negative control in the suite holds that open.
  defp apply_op(:block, [%Nx.Block.LogicalNot{}, [arg], _out | _], t, params, state) do
    {inp, state} = do_lower(arg, params, state)
    simple("Not", [inp], t, state)
  end

  # RSQRT IS TWO NODES: the specification has no Rsqrt, and Reciprocal(Sqrt(x)) is the form it
  # leaves to the caller. Read out of `onnx.defs` at opset 17 rather than assumed.
  defp apply_op(:rsqrt, [arg], t, params, state) do
    {inp, state} = do_lower(arg, params, state)
    {root, state} = simple("Sqrt", [inp], t, state)
    simple("Reciprocal", [root], t, state)
  end

  # NOT_EQUAL IS TWO NODES, because the specification does not carry it. Read out of `onnx.defs`
  # at opset 17 rather than assumed: Less, Greater, LessOrEqual, GreaterOrEqual, Equal and Not
  # are all present and NotEqual is not, so the negation is left to the caller. Every other
  # comparison is one node and sits in @binary above.
  defp apply_op(:not_equal, [lhs, rhs], t, params, state) do
    {l, state} = do_lower(lhs, params, state)
    {r, state} = do_lower(rhs, params, state)
    {eq, state} = simple("Equal", [l, r], t, state)
    simple("Not", [eq], t, state)
  end

  defp apply_op(:select, [pred, on_true, on_false], t, params, state) do
    {p, state} = do_lower(pred, params, state)
    {a, state} = do_lower(on_true, params, state)
    {b, state} = do_lower(on_false, params, state)
    simple("Where", [p, a, b], t, state)
  end

  # A CAST THAT CHANGES VALUES IS REFUSED, because the target accepts it and then does not
  # perform it. Measured against the Dataflow Compiler's own emulator, a f32 -> s32 -> f32
  # round trip parsed cleanly and came back untruncated, disagreeing with Nx by 0.875. A
  # silently wrong number is worse than a refusal. Casts that only relabel -- float to float,
  # int to int -- do not change values and are emitted, which is what a real export's Cast
  # nodes are.
  defp apply_op(:as_type, [arg], t, params, state) do
    from = Nx.type(arg)
    to = Nx.type(t)

    if lossy_cast?(from, to) do
      raise ArgumentError,
            "refusing to emit a Cast from #{inspect(from)} to #{inspect(to)}: the target " <>
              "accepts a value-changing Cast and does not apply it, so the graph would " <>
              "compute something other than the function it came from"
    end

    {inp, state} = do_lower(arg, params, state)
    {name, state} = fresh("Cast", state)

    node = %Node{
      input: [inp],
      output: [name],
      name: name,
      op_type: "Cast",
      attribute: [%Attribute{name: "to", type: :INT, i: onnx_type(to)}]
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
          "NxShuttle.Lowering has no ONNX lowering for Nx operation #{inspect(op)}"
  end

  defp lossy_cast?({:f, _}, {k, _}) when k in [:s, :u], do: true
  defp lossy_cast?({:s, a}, {:s, b}), do: b < a
  defp lossy_cast?({:u, a}, {:u, b}), do: b < a
  defp lossy_cast?({:s, _}, {:u, _}), do: true
  defp lossy_cast?({:u, _}, {:s, _}), do: true
  defp lossy_cast?({:f, a}, {:f, b}), do: b < a
  defp lossy_cast?(_, _), do: false

  # One arithmetic predicate: 1 where the input is nonzero, 0 where it is not. Used by
  # logical_and; kept separate because any other boolean lowering will want the same shape.
  defp nonzero_mask(input, t, state) do
    {a, state} = simple("Abs", [input], t, state)
    {big, state} = scalar_const(1.0e6, t, state)
    {scaled, state} = simple("Mul", [a, big], t, state)
    {eps, state} = scalar_const(-1.0e-6, t, state)
    {lo, state} = simple("Max", [scaled, eps], t, state)
    {one, state} = scalar_const(1.0, t, state)
    simple("Min", [lo, one], t, state)
  end

  # A broadcast scalar, emitted the same way the :constant clause does it.
  defp scalar_const(value, t, state) do
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
