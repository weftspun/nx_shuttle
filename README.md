# nx_shuttle

Compiles an `Nx.Defn` function to an ONNX graph.

A shuttle carries the weft thread across the warp. This carries a graph across to another
runtime: you write the arithmetic once in Nx, and it comes out as something an accelerator
toolchain will take.

It **compiles, it does not execute**. There is no backend here and no `Nx.default_backend/1`
to point at it. You hand it a function and templates, it hands you a `Onnx.ModelProto`.

```elixir
f = fn x -> Nx.multiply(Nx.add(x, 1.0), 0.5) end

{:ok, model} = NxShuttle.to_model(f, [Nx.template({1, 8, 8, 3}, :f32)])
:ok = NxShuttle.export(f, [Nx.template({1, 8, 8, 3}, :f32)], "net.onnx")
```

## Why this exists

It began as a fork of [axon_onnx](https://github.com/elixir-nx/axon_onnx), to add the operators
a real accelerator export needs. That turned out to be the wrong shape. **Axon's IR names
layers; ONNX names tensor operations**, so a layer whose body is an Nx function is opaque to
layer-level dispatch — and the operators below layer granularity are most of them.

Measured against a real 576px keypoint-detector device half, `num_windows=1`, the configuration
the Dataflow Compiler accepts: **825 nodes over 22 distinct operators**. Upstream's serializer
emitted four of them — `Constant`, `Conv`, `Sigmoid`, `Softmax`. The other eighteen had no
route at all.

So `NxShuttle.Lowering` traces the function to `Nx.Defn.Expr` and walks the expression instead.
The mapping is close to one-to-one, which is what makes it tractable: `:select` is `Where`,
`:as_type` is `Cast`, `:broadcast` is `Expand`, `:dot` is `MatMul`. Axon is not a dependency.

What survives from upstream is `lib/onnx/` — the ONNX protobuf schema as generated Protox
modules. That is the durable part, and the reason this stays a fork rather than a fresh
repository: the provenance of generated code should be visible.

## It refuses rather than guesses

A lowering that emits a plausible wrong node is worse than one that stops, so it stops:

- an Nx operation with no lowering is reported, naming the operation;
- a `dot` that is not a trailing/leading contraction is refused rather than emitted as a
  `MatMul` that would quietly mean something else;
- a **value-changing `Cast` is refused**. Measured: an `f32 → s32 → f32` round trip parsed
  cleanly on the accelerator toolchain and came back **untruncated**, disagreeing with Nx by
  0.875. It said yes and then computed something else. Relabelling casts still emit.

## Deployment targets

Two, and they are not interchangeable:

1. **The accelerator toolchain** (primary) — a Dataflow Compiler in a container. Authoritative,
   because it is the consumer that has to accept the graph.
2. **ONNX Runtime via [ortex](https://github.com/elixir-nx/ortex)** (backup) — self-contained
   NIF, no Python, no GIL.

Both run in the test suite and are **cross-checked against each other**, because the defect
that matters most is the one where they differ. Measured, eight operators on both targets:

| operator | primary | backup |
| --- | --- | --- |
| `add` `mul` `sub` `div` `add_mul` `sqrt` | accepted | accepted |
| `erf` | **refused** | accepted |
| `where` | **refused** | accepted |

`erf` and `where` are **not portable**: a model using them runs on the backup and never reaches
the accelerator. The suite prints that table rather than leaving it to be discovered when the
primary path is the one being shipped.

#### Supported, accepted, and folded are three different things

The compiler documents its operators in **two** tables, and reading only the first is a
mistake worth naming because it produced two wrong conclusions here. Table 3 is layers; Table 4
is activations. `Sqrt` and `Sigmoid` are absent from Table 3 and present in Table 4, so they
are supported and a Table-3-only reading calls them unsupported.

Worse, some operators are documented **only as a pattern**. `Erf` appears in neither table
alone; Table 4 reaches it through `Gelu (preview)`, whose ONNX form is `Mul, Erf`. So a bare
`Erf` is refused and an `Erf` inside a GELU is compiled -- which is why a real export contains
twelve of them and the one this suite emits standalone is rejected. The same holds for `SiLU`
(`Mul, Sigmoid`), hard-swish and Mish.

And a third category exists that no table describes. Comparing a passing 825-node export
against both tables, **twelve operators pass while being documented nowhere**:

    Cast Constant ConstantOfShape Expand Flatten Gather Identity Range
    Shape Squeeze Unsqueeze Where

These are shape plumbing, and they pass because a fixed-resolution graph **folds them away**
before anything reaches hardware -- not because they are implemented. That is a fragile kind of
support: `Where` is in that list from a real export, and the standalone `Where` this suite
emits is refused, because there is nothing to fold it into. `Cast` is the sharpest case,
accepted and then not performed.

So the rule this package works to: emit what the tables document, prefer the pattern form where
one exists, and treat "it compiled once" as weaker evidence than "it is in a table".
`weftspun/rf-detr-cpp`'s `scripts/hailo_supported_ops.py` carries both tables and computes
this delta.

### The primary is measured after quantization, not before

This is the part that is easy to get wrong. The compiler's `SDK_NATIVE` context runs the parsed
graph **in float**, and it agrees with Nx bit-exactly. That is not what deploys. The accelerator
runs a **quantized** graph, and the two are not the same number:

| operator | float emulation | quantized (deploys) |
| --- | --- | --- |
| `add` | 0.0 | 0.019428 |
| `sub` | 0.0 | 0.020283 |
| `mul` | 0.0 | 0.004498 |
| `div` | 0.0 | 0.004498 |
| `add_mul` | 0.0 | 0.009714 |
| `sqrt` | 0.0 | 0.008028 |

So the suite reports both. The float figure says whether the lowering is structurally right and
is asserted tightly; the quantized figure says what deployment will compute and is reported
rather than asserted, because quantization cost is a property of the target, not a defect here.

**Calibration is derived from the data, and getting this wrong is expensive.** A first version
calibrated over a fixed 0..2 while the inputs reached 4.375, so everything above 2 saturated
and it reported up to **2.375** error on a plain `Add` — a hundred times the real figure, and
entirely an artefact of the harness. The range now comes from the input's own min and max.
1024 entries, which is what the optimizer asks for: below that it drops to optimization level 0
and says so, and a number measured there is not one you would ship.

### Quantization is reducible, but not by anything the emitter can do

The residual above is **zero-mean rounding noise, not a systematic bias** — measured across six
operators, `|mean|` is 2% to 35% of the standard deviation, and none is bias-dominated. So a
correction folded into the graph at emission has nothing to cancel. We control both sides and
it does not help; the error is dither, not offset.

What does move it is the compiler's precision mode, measured on `(x + 1) * 0.5 - 0.25`:

| configuration | max residual | vs baseline |
| --- | --- | --- |
| `a8_w8` (default) | 0.008946 | — |
| **`a16_w16` whole model** | **0.000071** | **126x** |
| selective, `p/normalization1` | 0.004537 | 2.0x |
| selective, `p/output_layer1` | 0.005935 | 1.5x |

Two things worth knowing before reaching for it. **Selective 16-bit is not a cheap substitute**:
the layer carrying most of the error, `p/mul_and_add1`, appears in `get_hn_dict()` but
`quantization_param` reports it not found even fully qualified, and the wildcard form is
rejected by the script parser. And **`a16_w8` and `a16_w4` do not work here** — the 5.3.0
release notes announce them as preview for Hailo-10H, and the compiler rejects both with
`Unsupported value [PrecisionMode.a16_w8] for fields ['precision_mode']`. The mode that works
is `a16_w16` applied whole-model through
`model_optimization_config(compression_params, auto_16bit_weights_ratio=1)`.

Whether 16-bit is affordable is a throughput question rather than a memory one. The module
carries 4-8 GB of LPDDR4 and the accelerator has direct DRAM access, so a backbone whose
weights are ~25 MB at int8 and ~51 MB at int16 occupies about 1% of the smallest
configuration. The cost lands on bandwidth -- LPDDR4 32-bit at 4266 MT/s is about 17 GB/s, so
doubling the bytes halves a streaming-bound ceiling -- and on compute, where the datasheet
publishes 40 INT4 TOPS and the compiler guide 20 TOPS at 8-bit and **neither publishes a
16-bit figure**.

## Opset

Defaults to **17**, chosen by measurement rather than from a table. Neither consumer
distinguishes one opset from another:

    onnxruntime 1.29.0   14 operator cases, opsets 13..24     identical, every cell
    Dataflow Compiler    elementwise rank-4 graph             parsed at 13..23
    ortex (bundled ORT)  fixed at build time by the ort crate 21 loads, 22 does not

The compiler parsed 22 and 23 as readily as 15 although its guide documents 15–21, so the
published range is conservative rather than enforced. What it rejects is **operators, not
versions**.

The choice being free, the tie goes to the lowest opset that still expresses the whole operator
set, since a lower opset is readable by strictly more consumers. That is 17, where
`LayerNormalization` becomes an operator rather than a decomposition. Without it the answer
would be 15.

`NxShuttle.Reference.opset_ceiling/1` records both targets' ceilings, so a graph that would
deploy to the accelerator but not to the backup fails a check instead of surfacing later.

## IR version

Emitted at **ir_version 10**, pinned rather than inherited. The accelerator toolchain's bundled
checker refuses anything higher, and the current `onnx` python package defaults to 13 -- so a
graph built with library defaults is rejected before any operator is examined, with an error
about the checker rather than about the graph. That is not a hypothetical: it is how the
spec-versus-device equivalence check first failed.

## Rank and layout

ONNX is NCHW by convention and the compiler works in NHWC, so the test harness transposes on
the way in and back on the way out and compares in Nx's own layout. Rank-4 throughout: a rank-2
graph is rejected on its shape with a bare `IndexError`, which reads like a lowering fault and
is not one.

## Running the tests

```sh
mix test
```

The primary target needs a Dataflow Compiler image, which requires a licensed wheel that is not
redistributable and is in no repository. **An absent target is a failure, not a skip** — the
suite names what is missing and where to build it, because a run that quietly compares nothing
reports the same green as one that compared everything. If only one target is available it says
so, and says what that weaker run did and did not prove.

## Protobuf

`lib/onnx/` is generated with [protox](https://github.com/ahamez/protox). To regenerate:

```sh
mix generate_protobuf
```

You will need `protoc` (>= 3.0) on `$PATH`.

## Provenance

A fork of [axon_onnx](https://github.com/elixir-nx/axon_onnx) by Sean Moriarity, taken at
`a9256dd`. That commit is an ancestor of everything here. `lib/onnx/` is upstream's generated
protobuf and is unchanged; the Axon serializer and deserializer are removed, and the Nx
lowering replaces them.

GitHub records the fork's parent as `mortont/axon_onnx`, the root of the network, rather than
the `elixir-nx` repository this was actually taken from. The `upstream` remote and this
paragraph are where that is written down.

## License

Copyright (c) 2021 Sean Moriarity, and contributors to this fork.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions and
limitations under the License.
