defmodule NxShuttle do
  @moduledoc """
  Compiles an `Nx.Defn` function to an ONNX graph.

  This compiles; it does not execute. `Nx.Defn.Expr` is traced and lowered to ONNX nodes by
  `NxShuttle.Lowering`, and the result is a `Onnx.ModelProto` you can write to disk and hand to
  a downstream compiler or runtime.

      iex> f = fn x, y -> Nx.add(x, y) end
      iex> {:ok, model} = NxShuttle.to_model(f, [Nx.template({2}, :f32), Nx.template({2}, :f32)])
      iex> length(model.graph.node)
      1

  ## Opset

  Defaults to 17, and the reason is expressiveness rather than acceptance. Both consumers
  were swept and neither distinguishes one opset from another:

      onnxruntime 1.29.0   14 operator cases, opsets 13..24   identical, every cell
      Hailo DFC 5.3.0      elementwise rank-4 graph           PARSED at 13..23

  The DFC parsed 22 and 23 as readily as 15, though its guide documents 15-21, so the
  published range is conservative rather than enforced. What the DFC rejects is operators,
  not versions: the same graph carrying a standalone `Erf` is refused at every opset with
  `UnsupportedActivationLayerError`, even though a real rf-detr export contains twelve of
  them inside GELU patterns the parser fuses.

  So the choice is free, and the tie goes to the lowest opset that can still express the
  whole operator set, because a lower opset is readable by strictly more consumers and
  nothing measured is bought by going higher. That is 17, where `LayerNormalization` becomes
  an operator rather than a decomposition. Without it the answer would be 15.
  """

  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  @default_opset 17
  @ir_version 8

  @doc """
  Builds an `Onnx.ModelProto` from `fun` applied to `templates`.

  Returns `{:ok, model}`, or `{:error, reason}` when an operation has no lowering.

  ## Options

    * `:opset` - ONNX opset version, defaults to #{@default_opset}
    * `:names` - input names, defaults to `input_0`, `input_1`, ...
    * `:doc_string` - graph doc string
  """
  def to_model(fun, templates, opts \\ []) do
    opset = opts[:opset] || @default_opset
    names = opts[:names] || Enum.map(0..(length(templates) - 1)//1, &"input_#{&1}")

    expr = apply(Nx.Defn.debug_expr(fun), templates)
    params = names |> Enum.with_index() |> Map.new(fn {n, i} -> {i, n} end)

    {out, nodes, inits, _state} = NxShuttle.Lowering.lower(expr, params)

    graph = %Graph{
      node: Enum.reverse(nodes),
      name: opts[:name] || "nx_shuttle",
      doc_string: opts[:doc_string] || "",
      input: Enum.zip_with(names, templates, &value_info/2),
      output: [value_info(out, expr)],
      initializer: Enum.reverse(inits)
    }

    {:ok,
     %Model{
       ir_version: @ir_version,
       producer_name: "nx_shuttle",
       producer_version: Application.spec(:nx_shuttle, :vsn) |> to_string(),
       opset_import: [%Opset{domain: "", version: opset}],
       graph: graph
     }}
  rescue
    e in ArgumentError -> {:error, Exception.message(e)}
  end

  @doc """
  Same as `to_model/3` but raises on failure.
  """
  def to_model!(fun, templates, opts \\ []) do
    case to_model(fun, templates, opts) do
      {:ok, model} -> model
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc """
  Writes the compiled graph to `path`.
  """
  def export(fun, templates, path, opts \\ []) do
    with {:ok, model} <- to_model(fun, templates, opts) do
      File.write(path, Model.encode!(model))
    end
  end

  @doc """
  Encodes a model to its ONNX wire bytes.

  `Onnx.ModelProto.encode!/1` returns iodata; this flattens it, because anything that is not
  `File.write/2` -- a NIF boundary, a socket, a checksum -- wants one binary.
  """
  def encode(%Model{} = model), do: model |> Model.encode!() |> IO.iodata_to_binary()

  defp value_info(name, tensor) do
    dims =
      tensor |> Nx.shape() |> Tuple.to_list() |> Enum.map(&%Dimension{value: {:dim_value, &1}})

    %Value{
      name: name,
      type: %Type{
        value:
          {:tensor_type,
           %Placeholder{
             elem_type: NxShuttle.Lowering.onnx_type(Nx.type(tensor)),
             shape: %Shape{dim: dims}
           }}
      }
    }
  end
end
