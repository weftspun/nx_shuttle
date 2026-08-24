defmodule NxShuttle.LoweringTest do
  use ExUnit.Case, async: false

  # THE REFERENCE IS THE COMPILER THAT HAS TO ACCEPT THE GRAPH, not a general-purpose runtime.
  # A runtime agreeing with Nx says only that the bytes are well formed; it says nothing about
  # whether the graph reaches the accelerator, which is the question this library exists to
  # answer. So the DFC parses each graph and its native emulator runs it, and the number it
  # returns is what the lowering is measured against.

  alias NxShuttle.DFC

  # ONNX is NCHW by convention and the DFC works in NHWC, so every case is rank-4 and the
  # transpose happens in DFC.run/1. A rank-2 graph is rejected on its shape, which would read
  # as a lowering fault and is not one.
  @x Nx.iota({1, 4, 4, 2}, type: :f32) |> Nx.divide(8.0) |> Nx.add(0.5)

  setup_all do
    Nx.default_backend(Torchx.Backend)

    # An unmet precondition is a FAIL, not a skip. A suite that quietly runs zero comparisons
    # reports the same green as one that ran them all, and this reference is the kind that goes
    # missing: it needs a licensed wheel and a multi-gigabyte image.
    case DFC.available() do
      :ok ->
        :ok

      {:error, why} ->
        raise """
        The Hailo Dataflow Compiler is not runnable here, so nothing can be measured: #{why}

        Build it with deploy/hailo-dfc/build.sh in weftspun/rf-detr-cpp; it needs the licensed
        wheel from hailo.ai, which is not redistributable and is not in any repository.
        """
    end
  end

  defp compile_all(cases) do
    tpl = [Nx.template(Nx.shape(@x), Nx.type(@x))]

    cases
    |> Enum.map(fn {name, fun} ->
      %{name: name, model: NxShuttle.encode(NxShuttle.to_model!(fun, tpl)), input: @x}
    end)
    |> DFC.run()
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
      {"where", &Nx.select(Nx.equal(&1, &1), Nx.multiply(&1, 2.0), &1)}
    ]
  end

  describe "what the Dataflow Compiler accepts, and whether it computes what Nx does" do
    test "each operator, against the compiler's own emulator" do
      cs = cases()
      results = compile_all(cs)
      funs = Map.new(cs)

      rows =
        Enum.map(cs, fn {name, _} ->
          case results[name] do
            {:ok, got} ->
              expected = funs[name].(@x)

              diff =
                Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

              {name, if(diff <= 1.0e-5, do: :agrees, else: {:disagrees, diff}), diff}

            {:error, why} ->
              {name, :rejected, why}
          end
        end)

      IO.puts("\n  operator   verdict")

      Enum.each(rows, fn
        {n, :agrees, d} -> IO.puts("  #{String.pad_trailing(n, 10)} agrees (max|diff| #{d})")
        {n, {:disagrees, d}, _} -> IO.puts("  #{String.pad_trailing(n, 10)} DISAGREES by #{d}")
        {n, :rejected, why} -> IO.puts("  #{String.pad_trailing(n, 10)} rejected: #{String.slice(why, 0, 88)}")
      end)

      # Operators the compiler refuses are a fact about the target, not a defect here, and they
      # are named rather than counted as passes. What must never happen is a graph it accepts
      # and then computes differently from Nx: that is the lowering being wrong.
      wrong = for {n, {:disagrees, d}, _} <- rows, do: "#{n} by #{d}"
      assert wrong == [], "the compiler accepted these and computed something else: #{inspect(wrong)}"

      accepted = for {n, :agrees, _} <- rows, do: n
      assert accepted != [], "the compiler accepted nothing, so nothing was measured"
      IO.puts("\n  #{length(accepted)}/#{length(cs)} accepted and agreeing: #{Enum.join(accepted, " ")}")
    end
  end

  describe "negative controls" do
    test "a subtraction lowered with its operands swapped must NOT agree" do
      # Sub is not commutative, so a reversed lowering is exactly the class of bug the
      # agreement test above exists to catch. If this passes, that test is decoration.
      tpl = [Nx.template(Nx.shape(@x), Nx.type(@x))]
      swapped = fn t -> Nx.subtract(Nx.multiply(t, 0.0) |> Nx.add(0.25), t) end

      results =
        DFC.run([
          %{name: "swapped", model: NxShuttle.encode(NxShuttle.to_model!(swapped, tpl)), input: @x}
        ])

      case results["swapped"] do
        {:ok, got} ->
          expected = Nx.subtract(@x, 0.25)
          diff = Nx.subtract(got, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

          assert diff > 1.0e-3,
                 "a swapped subtraction agreed to #{diff}; the comparison cannot see operand order"

        {:error, why} ->
          flunk("the control could not run, so it proves nothing: #{why}")
      end
    end

    test "an Nx operation with no lowering is reported, not silently dropped" do
      tpl = [Nx.template({1, 4, 4, 2}, :f32)]
      assert {:error, message} = NxShuttle.to_model(&Nx.sort(&1, axis: 0), tpl)
      assert message =~ "no ONNX lowering for Nx operation"
    end

    test "a value-changing Cast is refused, because the target accepts and ignores it" do
      # Measured: f32 -> s32 -> f32 parsed and came back untruncated, off by 0.875. The
      # compiler said yes and then computed something else, which is the one failure mode a
      # parse-only check cannot see.
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

    test "a dot that is not a trailing/leading contraction is reported" do
      tpl = [Nx.template({2, 4}, :f32), Nx.template({2, 4}, :f32)]
      assert {:error, message} = NxShuttle.to_model(fn x, y -> Nx.dot(x, [0], y, [0]) end, tpl)
      assert message =~ "only a trailing/leading contraction"
    end
  end
end
