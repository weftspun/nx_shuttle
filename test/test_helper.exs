# The python side is DECLARED, which is the whole of the uv objection in CLAUDE.md. This
# pyproject is tracked, the versions are pinned, and `pythonx` resolves it into a locked
# environment -- so the reference runtime can be rebuilt on another desk rather than being
# whatever `pip install` last offered.
Pythonx.uv_init("""
[project]
name = "nx_onnx_reference"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "numpy==2.2.6",
  "onnx==1.20.0",
  "onnxruntime==1.29.0"
]
""")

ExUnit.start()
