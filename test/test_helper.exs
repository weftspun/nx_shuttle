# Quantizing a graph takes the optimizer well past ExUnit's default 60s: 1024 calibration
# entries per case, in a container. The timeout is raised rather than the calibration lowered,
# because below 1024 the optimizer drops to optimization level 0 and says so, and a
# quantization figure measured there is not the one deployment would see.
ExUnit.start(timeout: 600_000)
