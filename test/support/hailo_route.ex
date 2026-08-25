defmodule NxShuttle.HailoRoute do
  @moduledoc false

  # What happens to an operator when the full ONNX opset meets this accelerator.
  #
  # This is not the vendor's tables. It is the DECISION TREE those tables sit inside, because
  # the tables answer one question -- is it listed -- and the compiler answers four, and the
  # order the questions are asked in is what determines the outcome.
  #
  # Every branch below is measured. The user guide (5.3.0, Table 3 "Supported ONNX Layers" and
  # Table 4 "Supported Activations") predicts most of them and is wrong or silent about three,
  # which are marked.

  # THE FULL ai.onnx OPSET, 15-21. Generated from `onnx.defs` and the 5.3.0 user guide, not
  # typed by hand.
  #
  #   193  operators in the opset
  #    62  named in Table 3 (Supported ONNX Layers) or Table 4 (Supported Activations)
  #   131  named in neither
  #    35  MEASURED here, one graph per container run, against both engines
  #
  # The gap between 62 and 35 is the point of keeping provenance. A documented operator that has
  # not been run is a PREDICTION from the vendor's table, and the tables have already been wrong
  # once: Table 4 gives Gelu's ONNX form as `Mul, Erf`, and emitting exactly that is refused.
  @opset ~w(
    Abs Add ArgMax AveragePool BatchNormalization Clip Concat Conv ConvTranspose DepthToSpace
    Div Dropout Elu Equal Erf Exp GRU Gelu Gemm Greater HardSigmoid LSTM LayerNormalization
    LeakyRelu Less Log LogSoftmax MatMul Max MaxPool Mean Min Mish Mul Neg OneHot PRelu Pad Pow
    RNN ReduceL2 ReduceMax ReduceMean ReduceMin ReduceSum ReduceSumSquare Relu Reshape Resize
    Sigmoid Slice Softmax Softplus Softsign SpaceToDepth Split Sqrt Sub Sum Tanh Transpose
    Upsample Acos Acosh AffineGrid And ArgMin Asin Asinh Atan Atanh Bernoulli BitShift
    BitwiseAnd BitwiseNot BitwiseOr BitwiseXor BlackmanWindow Cast CastLike Ceil Celu
    CenterCropPad Col2Im Compress ConcatFromSequence Constant ConstantOfShape ConvInteger Cos
    Cosh CumSum DFT DeformConv DequantizeLinear Det DynamicQuantizeLinear Einsum Expand EyeLike
    Flatten Floor Gather GatherElements GatherND GlobalAveragePool GlobalLpPool GlobalMaxPool
    GreaterOrEqual GridSample GroupNormalization HammingWindow HannWindow HardSwish Hardmax
    Identity If ImageDecoder InstanceNormalization IsInf IsNaN LRN LessOrEqual Loop
    LpNormalization LpPool MatMulInteger MaxRoiPool MaxUnpool MeanVarianceNormalization
    MelWeightMatrix Mod Multinomial NegativeLogLikelihoodLoss NonMaxSuppression NonZero Not
    Optional OptionalGetElement OptionalHasElement Or QLinearConv QLinearMatMul QuantizeLinear
    RandomNormal RandomNormalLike RandomUniform RandomUniformLike Range Reciprocal ReduceL1
    ReduceLogSum ReduceLogSumExp ReduceProd RegexFullMatch ReverseSequence RoiAlign Round STFT
    Scan Scatter ScatterElements ScatterND Selu SequenceAt SequenceConstruct SequenceEmpty
    SequenceErase SequenceInsert SequenceLength SequenceMap Shape Shrink Sign Sin Sinh Size
    SoftmaxCrossEntropyLoss SplitToSequence Squeeze StringConcat StringNormalizer StringSplit
    Tan TfIdfVectorizer ThresholdedRelu Tile TopK Trilu Unique Unsqueeze Where Xor
  )

  @documented ~w(
    Abs Add ArgMax AveragePool BatchNormalization Clip Concat Conv ConvTranspose DepthToSpace
    Div Dropout Elu Equal Erf Exp GRU Gelu Gemm Greater HardSigmoid LSTM LayerNormalization
    LeakyRelu Less Log LogSoftmax MatMul Max MaxPool Mean Min Mish Mul Neg OneHot PRelu Pad Pow
    RNN ReduceL2 ReduceMax ReduceMean ReduceMin ReduceSum ReduceSumSquare Relu Reshape Resize
    Sigmoid Slice Softmax Softplus Softsign SpaceToDepth Split Sqrt Sub Sum Tanh Transpose
    Upsample
  )

  @measured %{
    "Abs" => :accept,
    "Add" => :accept,
    "Cast" => :refused,
    "Ceil" => :refused,
    "Concat" => :accept,
    "Cos" => :refused,
    "Div" => :accept,
    "Equal" => :accept,
    "Erf" => :refused,
    "Exp" => :accept,
    "Expand" => :refused,
    "Floor" => :accepted_but_ignored,
    "Greater" => :accept,
    "GreaterOrEqual" => :refused,
    "Less" => :accept,
    "LessOrEqual" => :refused,
    "Log" => :accept,
    "Max" => :accept,
    "Min" => :accept,
    "Mod" => :refused,
    "Mul" => :accept,
    "Neg" => :accept,
    "Not" => :refused,
    "Pow" => :accept,
    "Reciprocal" => :accept,
    "Reshape" => :refused,
    "Sigmoid" => :accept,
    "Sign" => :refused,
    "Sin" => :refused,
    "Slice" => :accept,
    "Sqrt" => :accept,
    "Sub" => :accept,
    "Tanh" => :accept,
    "Transpose" => :refused,
    "Where" => :refused
  }

  @doc """
  Every ai.onnx operator in 15-21, so a caller can route the whole opset rather than a guess.
  """
  def opset, do: @opset

  @doc """
  The 62 named in Table 3 or Table 4 of the 5.3.0 guide.
  """
  def documented, do: @documented

  @doc """
  The 35 run here, one graph per container, against both engines.
  """
  def measured, do: @measured

  @doc """
  Route an operator, in a domain, appearing in a graph of a given shape.

  Returns the decision and the reason, so a refusal can say which branch produced it.
  """
  def route(op), do: route(op, [])

  def route(op, opts) do
    domain = Keyword.get(opts, :domain, "")
    opset = Keyword.get(opts, :opset, 17)
    folds_to_nothing? = Keyword.get(opts, :folds_to_nothing, false)
    standalone? = Keyword.get(opts, :standalone, true)
    # The INPUT domain: the interval the tensor actually spans. Two separate things depend on it
    # and neither is visible in any support table.
    {lo, hi} = Keyword.get(opts, :input_domain, {0.5, 5.375})

    cond do
      # 1. DOMAIN, BEFORE ANYTHING ELSE. A custom domain has no lowering to hardware at all --
      # not a missing operator, a missing compilation path. onnxruntime-extensions puts
      # tokenizers and string operators under ai.onnx.contrib, and nothing there reaches a
      # device. The check is first because a listed operator NAME in the wrong domain is a
      # different operator.
      domain not in ["", "ai.onnx"] ->
        {:refuse, {:foreign_domain, domain}}

      # 2. OPSET RANGE. The guide states 15-21. Outside it the parser does not get as far as
      # looking at operators.
      opset < 15 or opset > 21 ->
        {:refuse, {:opset_out_of_range, opset, 15..21}}

      # 3. THE GRAPH, NOT THE OPERATOR. A graph that folds away is refused whatever it contained.
      # Measured: identity, add(x, 0.0) and mul(x, 1.0) all return UnsupportedModelError, while
      # add(x, 1.0) is exact; equal(t, t) returns "ValidationError: [] should be non-empty",
      # which is a complaint about an EMPTY NODE LIST.
      #
      # This branch exists because mistaking it for an operator refusal is exactly what happened:
      # equal(t, t) was read as "Equal is unsupported" and that claim reached several commits
      # before equal(t, const) was measured compiling.
      folds_to_nothing? ->
        {:refuse, :graph_folds_to_nothing}

      # 4. THE CALIBRATION DOMAIN, WHICH IS NOT THE INPUT DOMAIN, and no support table mentions
      # either. This branch was written against `lo` and was WRONG; the mixed-input sweep caught
      # it.
      #
      # The optimizer does not calibrate on the tensor. It calibrates on a PADDED interval --
      # `pad = 0.05 * max(|lo|, |hi|, 1)`, then draws 1024 samples from `[lo - pad, hi + pad]`.
      # So a strictly positive tensor can still hand negatives to a partial function, and the
      # wider the range the further the padding reaches:
      #
      #     input [0.5, 5.375]   pad 0.27   calibrates [0.23, 5.64]    sqrt exact
      #     input [0.125, 16.0]  pad 0.80   calibrates [-0.675, 16.8]  sqrt REFUSED RuntimeError
      #                                                                log  REFUSED AccelerasNegative
      #                                                                rsqrt REFUSED AccelerasNegative
      #
      # Both inputs are positive. Only one survives, and the difference is the padding.
      op in ~w(Log Sqrt Reciprocal) and calib_lo(lo, hi) <= 0.0 ->
        {:refuse_at_emission, {:calibration_reaches_negative, op, {lo, hi}, calib_lo(lo, hi)}}

      # And quantization is set by the OUTPUT span rather than by the operator. Measured on Exp:
      # over [0.5, 5.375] the output spans 1.65..79.44 and the error is 0.483141; over
      # [0.125, 16.0] it spans e^16 and the error is 0.25 on a far larger scale. Rewriting
      # exp(x) as exp(x/2)*exp(x/2) -- identical span -- moved it only to 0.445812, so the graph
      # shape is not the lever. A caveat rather than a refusal: it compiles and computes.
      op == "Exp" and exp_span_ratio(lo, hi) > 8.0 ->
        {:accept_with_caveat,
         {:quantization_span, exp_span_ratio(lo, hi), "~0.6% of output span at a8_w8"}}

      # 4. MEASURED, and it outranks the tables. These have been run here, one graph per
      # container, against the accelerator and the ONNX reference evaluator.
      Map.has_key?(@measured, op) ->
        measured_decision(op, standalone?)

      # 5. STRICT COMPARISONS ONLY. Table 4 lists Less, Greater and Equal and does not list
      # LessOrEqual, GreaterOrEqual or Not. `1 - strict` reaches them with operators that are
      # listed, and is measured exact on both engines.
      op in ~w(NotEqual) ->
        {:decompose, {:arithmetic_negation, "Equal"}}

      # 6. PATTERN-ONLY. Table 4 reaches Erf only through `Gelu (preview)`, ONNX form `Mul, Erf`.
      # Measured: emitting that form is ALSO refused, so the compiler recognising its own pattern
      # is not the same as writing the pattern out. The one place the guide misled us.
      op in ~w(HardSigmoid Softplus) ->
        {:refuse, {:pattern_only, pattern_for(op)}}

      # 7. DOCUMENTED BUT NEVER RUN HERE. 62 operators are named in Table 3 or Table 4 and 35
      # have been measured. This branch is the other 27, and it returns a PREDICTION rather than
      # a result -- kept distinct because the tables have been wrong before.
      op in @documented ->
        {:accept, :predicted_from_table}

      # 8. IN THE OPSET, NAMED IN NEITHER TABLE. 131 of 193. Predicted refusal; every one of this
      # class that has been measured -- Sign, Sin, Cos, Ceil, Mod, Not, Where -- was refused.
      op in @opset ->
        {:refuse, {:undocumented, op}}

      # 9. NOT AN ai.onnx OPERATOR AT ALL in 15-21.
      true ->
        {:refuse, {:not_in_opset, op}}
    end
  end

  # A measured operator's decision, with the input-domain checks applied first because an
  # operator can be measured working and still be wrong on the tensor you have.
  defp measured_decision(op, standalone?) do
    case Map.fetch!(@measured, op) do
      :accept ->
        {:accept, :measured}

      :accepted_but_ignored ->
        {:refuse_at_emission, {:accepted_but_ignored, op}}

      :refused ->
        if op in ~w(Reshape Transpose Expand Cast) and not standalone? do
          {:accept, :folded_in_context}
        else
          {:refuse, {:measured_refused, op}}
        end
    end
  end

  # The documented ONNX form each pattern-only operator is reachable through -- and the measured
  # fact that writing that form out is not sufficient.
  defp pattern_for("Erf"), do: {"Gelu (preview)", ~w(Mul Erf), :measured_still_refused}
  defp pattern_for("HardSigmoid"), do: {"Hard-swish (preview)", ~w(Mul HardSigmoid), :untested}
  defp pattern_for("Softplus"), do: {"Mish", ~w(Mul Tanh Softplus), :untested}

  defp negation_of("LessOrEqual"), do: "Greater"
  defp negation_of("GreaterOrEqual"), do: "Less"
  defp negation_of("Not"), do: :operand
  defp negation_of("NotEqual"), do: "Equal"

  # The ratio the quantizer has to cover. It is the OUTPUT span that matters, which for Exp is
  # e^hi / e^lo -- an input interval of five spans a factor of 148.
  defp exp_span_ratio(lo, hi), do: :math.exp(hi) / :math.exp(lo)

  # What the optimizer actually sees. `DFC.run/2` pads by 5% of the largest magnitude and draws
  # its 1024 calibration samples from the padded interval, so this -- not `lo` -- is the number
  # a partial function has to survive.
  defp calib_lo(lo, hi), do: lo - 0.05 * max(max(abs(lo), abs(hi)), 1.0)
end
