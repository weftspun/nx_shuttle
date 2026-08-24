defmodule NxShuttle.LoweringTest do
  use ExUnit.Case, async: false

  # THE REFERENCE IS THE COMPILER THAT HAS TO ACCEPT THE GRAPH, not a general-purpose runtime.
  # A runtime agreeing with Nx says only that the bytes are well formed; it says nothing about
  # whether the graph reaches the accelerator, which is the question this library exists to
  # answer. So the DFC parses each graph and its native emulator runs it, and the number it
  # returns is what the lowering is measured against.

  alias NxShuttle.Reference

  # ONNX is NCHW by convention and the DFC works in NHWC, so every case is rank-4 and the
  # transpose happens in DFC.run/1. A rank-2 graph is rejected on its shape, which would read
  # as a lowering fault and is not one.
  @x Nx.iota({1, 4, 4, 2}, type: :f32) |> Nx.divide(8.0) |> Nx.add(0.5)

  setup_all do
    Nx.default_backend(Torchx.Backend)

    # An unmet precondition is a FAIL, not a skip. A suite that quietly runs zero comparisons
    # reports the same green as one that ran them all, and this reference is the kind that goes
    # missing: it needs a licensed wheel and a multi-gigabyte image.
    {engines, missing} = Reference.engines()

    if engines == [] do
      raise """
      Nothing can be measured here. No deployment target is runnable:

      #{Enum.map_join(missing, "
", fn {e, why} -> "  #{e}: #{why}" end)}

      The compiler is the primary target; build it with deploy/hailo-dfc/build.sh in
      weftspun/rf-detr-cpp. It needs the licensed wheel from hailo.ai, which is not
      redistributable and is in no repository.
      """
    end

    IO.puts("
  " <> Reference.caveat(engines))
    Enum.each(missing, fn {e, why} -> IO.puts("  not run: #{e} -- #{why}") end)
    {:ok, engines: engines}
  end

  defp fmt(f) when is_float(f), do: :erlang.float_to_binary(f, [{:decimals, 6}, :compact])
  defp fmt(other), do: inspect(other)

  defp compile_all(engines, cases) do
    tpl = [Nx.template(Nx.shape(@x), Nx.type(@x))]

    built =
      Enum.map(cases, fn {name, fun} ->
        %{name: name, model: NxShuttle.encode(NxShuttle.to_model!(fun, tpl)), input: @x}
      end)

    Map.new(engines, fn e -> {e, Reference.run(e, built)} end)
  end

  defp cases do
    [
      {"add", &Nx.add(&1, 1.0)},
      {"mul", &Nx.multiply(&1, 0.5)},
      {"sub", &Nx.subtract(&1, 0.25)},
      {"div", &Nx.divide(&1, 2.0)},
      {"add_mul", &Nx.multiply(Nx.add(&1, 1.0), 0.5)},
      {"sqrt", &Nx.sqrt/1},
      {"erf", &Nx.erf/1},
      {"where", &Nx.select(Nx.equal(&1, &1), Nx.multiply(&1, 2.0), &1)},
      # DOES THE RESIDUAL ACCUMULATE? Each link is arithmetically the identity, so a chain of
      # them has the same float answer as one. Anything the quantized graph adds along the way
      # is the noise compounding, and depth is the axis nobody measures on a single operator.
      {"chain1", &chain(&1, 1)},
      {"chain2", &chain(&1, 2)},
      {"chain4", &chain(&1, 4)},
      {"chain8", &chain(&1, 8)}
    ]
  end

  # RETRACTED AND REPLACED. The first version chained affine links -- (x+1)*0.5*2-1 -- and
  # reported byte-identical residuals at depths 1, 2, 4 and 8. That was not noise cancelling,
  # it was the compiler FOLDING the chain into the same two-layer graph, and the flat line was
  # evidence about the optimizer rather than about accumulation.
  #
  # `sqrt` is not affine, so the folder cannot collapse these. Each link is a real quantization
  # boundary and depth is finally the thing being varied.
  defp chain(x, n) do
    Enum.reduce(1..n//1, x, fn _, acc ->
      acc |> Nx.multiply(acc) |> Nx.add(0.01) |> Nx.sqrt()
    end)
  end

  describe "every deployment target, and whether they agree with each other" do
    test "each operator, on each target", %{engines: engines} do
      cs = cases()
      per_engine = compile_all(engines, cs)
      funs = Map.new(cs)

      rows =
        for engine <- engines, {name, _} <- cs do
          expected = funs[name].(@x)

          d = fn t ->
            Nx.subtract(t, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          end

          verdict =
            case per_engine[engine][name] do
              # The primary reports twice: the float emulation says whether the lowering is
              # structurally right, the quantized one says what deployment will actually
              # compute. They are different questions and get different tolerances.
              {:ok, %{native: native, bit_exact: exact}} ->
                # Decompose the quantization residual. A constant folded into the graph can
                # only cancel the MEAN; the spread around it is rounding noise and no emission
                # choice removes it. Reporting max alone cannot tell those apart.
                r = Nx.subtract(exact, expected)
                mean = r |> Nx.mean() |> Nx.to_number()
                sd = r |> Nx.standard_deviation() |> Nx.to_number()
                {:two, d.(native), d.(exact), mean, sd}

              {:ok, got} ->
                diff = d.(got)
                if diff <= 1.0e-5, do: {:agrees, diff}, else: {:disagrees, diff}

              {:error, why} ->
                {:rejected, why}
            end

          {engine, name, verdict}
        end

      IO.puts("
  target  operator   verdict")

      Enum.each(rows, fn {e, n, v} ->
        text =
          case v do
            {:two, nat, exact, mean, sd} ->
              share = if exact > 0.0, do: abs(mean) / exact * 100.0, else: 0.0

              "float #{fmt(nat)} | quantized max #{fmt(exact)} bias #{fmt(mean)} " <>
                "sd #{fmt(sd)} (bias is #{fmt(share)}% of max)"

            {:agrees, d} ->
              "agrees (max|diff| #{fmt(d)})"

            {:disagrees, d} ->
              "DISAGREES by #{fmt(d)}"

            {:rejected, why} ->
              "rejected: #{String.slice(why, 0, 60)}"
          end

        IO.puts("  #{String.pad_trailing("#{e}", 7)} #{String.pad_trailing(n, 10)} #{text}")
      end)

      # A target refusing an operator is a fact about that target, named rather than counted as
      # a pass. What must never happen is a target ACCEPTING a graph and computing something
      # else: that is the lowering being wrong, and it is the one thing a parse check misses.
      # The lowering is wrong only if the FLOAT path disagrees. Quantization error is a
      # property of the target and is reported, not asserted away.
      wrong =
        for {e, n, v} <- rows,
            d =
              case(v,
                do: (
                  {:disagrees, x} -> x
                  {:two, x, _, _, _} when x > 1.0e-5 -> x
                  _ -> nil
                )
              ),
            d != nil,
            do: "#{e}/#{n} by #{d}"

      assert wrong == [], "accepted and computed something else in float: #{inspect(wrong)}"

      quantization =
        for {_e, n, {:two, _nat, exact, mean, sd}} <- rows, do: {n, exact, mean, sd}

      if quantization != [] do
        {n, mx, mean, sd} = Enum.max_by(quantization, &elem(&1, 1))
        IO.puts("
  worst quantization residual: #{n}, max #{fmt(mx)}, bias #{fmt(mean)}, sd #{fmt(sd)}")

        correctable =
          Enum.count(quantization, fn {_n, _mx, m, s} -> s > 0.0 and abs(m) > 0.5 * s end)

        IO.puts(
          "  #{correctable}/#{length(quantization)} residuals are bias-dominated (|mean| > sd/2), " <>
            "which is the part a folded correction could remove"
        )
      end

      accepted =
        for {e, n, v} <- rows,
            match?({:agrees, _}, v) or match?({:two, _, _, _, _}, v),
            do: {e, n}

      assert accepted != [], "no target accepted anything, so nothing was measured"

      # THE CROSS-CHECK, and the reason both targets run rather than whichever was handy. An
      # operator one target deploys and the other refuses means the model runs on the primary
      # and not the backup, or the reverse. That is a portability fact, and it should be read
      # off a table rather than discovered when the backup is needed.
      if length(engines) > 1 do
        split =
          for {n, rs} <- Enum.group_by(rows, fn {_e, n, _v} -> n end),
              ok = for({e, _, {:agrees, _}} <- rs, do: e),
              no = for({e, _, {:rejected, _}} <- rs, do: e),
              ok != [] and no != [],
              do: "#{n}: deploys on #{inspect(ok)}, refused by #{inspect(no)}"

        if split != [] do
          IO.puts("
  NOT PORTABLE across targets:")
          Enum.each(split, &IO.puts("    " <> &1))
        end
      end

      IO.puts("
  #{length(accepted)} target/operator pairs agreeing")
    end
  end

  describe "negative controls" do
    test "a subtraction with its operands swapped must NOT agree", %{engines: engines} do
      # Sub is not commutative, so a reversed lowering is exactly the class of bug the
      # agreement test exists to catch. If this passes, that test is decoration.
      tpl = [Nx.template(Nx.shape(@x), Nx.type(@x))]
      swapped = fn t -> Nx.subtract(Nx.multiply(t, 0.0) |> Nx.add(0.25), t) end

      built = [
        %{name: "swapped", model: NxShuttle.encode(NxShuttle.to_model!(swapped, tpl)), input: @x}
      ]

      expected = Nx.subtract(@x, 0.25)

      checked =
        for engine <- engines, {:ok, res} <- [Reference.run(engine, built)["swapped"]] do
          got = if is_map(res) and is_map_key(res, :native), do: res.native, else: res
          diff = Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

          assert diff > 1.0e-3,
                 "#{engine} agreed to #{diff} on a swapped subtraction; it cannot see operand order"

          engine
        end

      assert checked != [], "no target ran the control, so it proves nothing"
    end

    test "a value-changing Cast is refused, because a target accepts and ignores it" do
      # Measured: f32 -> s32 -> f32 parsed on the accelerator toolchain and came back
      # untruncated, off by 0.875. It said yes and then computed something else.
      tpl = [Nx.template({1, 4, 4, 2}, :f32)]

      assert {:error, message} =
               NxShuttle.to_model(&(Nx.as_type(&1, {:s, 32}) |> Nx.as_type({:f, 32})), tpl)

      assert message =~ "refusing to emit a Cast"
    end

    test "a relabelling Cast is still emitted" do
      tpl = [Nx.template({1, 4, 4, 2}, :f32)]
      assert {:ok, model} = NxShuttle.to_model(&Nx.as_type(&1, {:f, 32}), tpl)
      assert is_struct(model, Onnx.ModelProto)
    end

    test "an Nx operation with no lowering is reported, not silently dropped" do
      tpl = [Nx.template({1, 4, 4, 2}, :f32)]
      assert {:error, message} = NxShuttle.to_model(&Nx.sort(&1, axis: 0), tpl)
      assert message =~ "no ONNX lowering for Nx operation"
    end

    test "a dot that is not a trailing/leading contraction is reported" do
      tpl = [Nx.template({2, 4}, :f32), Nx.template({2, 4}, :f32)]
      assert {:error, message} = NxShuttle.to_model(fn x, y -> Nx.dot(x, [0], y, [0]) end, tpl)
      assert message =~ "only a trailing/leading contraction"
    end

    test "when a target is missing, it is named rather than assumed absent" do
      # A fallback nothing ever takes is a rule nobody has verified. Pointing at an image that
      # does not exist forces the miss.
      System.put_env("NX_SHUTTLE_DFC_IMAGE", "nx-shuttle-no-such-image:never")

      try do
        {engines, missing} = Reference.engines()
        refute :dfc in engines, "the compiler was selected despite a nonexistent image"
        assert {:dfc, why} = List.keyfind(missing, :dfc, 0)
        assert why =~ "is not built"
      after
        System.delete_env("NX_SHUTTLE_DFC_IMAGE")
      end
    end
  end
end
