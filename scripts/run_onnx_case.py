"""Run a model against TensorProto inputs and write the output as a TensorProto.

Usage: run_onnx_case.py <model.onnx> <out.pb> <in0.pb> [<in1.pb> ...]

The model is checked before it is run, so a graph that ONNX rejects fails here rather
than producing a number the caller would compare.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def main():
    model_path, out_path, in_paths = sys.argv[1], sys.argv[2], sys.argv[3:]

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    inputs = {}
    for name, path in zip([i.name for i in model.graph.input], in_paths):
        with open(path, "rb") as fh:
            tp = onnx.TensorProto()
            tp.ParseFromString(fh.read())
        inputs[name] = numpy_helper.to_array(tp)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, inputs)[0]

    with open(out_path, "wb") as fh:
        fh.write(numpy_helper.from_array(np.ascontiguousarray(out)).SerializeToString())


if __name__ == "__main__":
    main()
