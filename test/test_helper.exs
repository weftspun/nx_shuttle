# Quantizing a graph takes the optimizer well past ExUnit's default 60s: 1024 calibration
# entries per case, in a container. The timeout is raised rather than the calibration lowered,
# because below 1024 the optimizer drops to optimization level 0 and says so, and a
# quantization figure measured there is not the one deployment would see.
#
# 1_800_000 rather than 600_000, and the number is measured rather than guessed. Twelve cases
# on this desk (Windows, 16 logical cores, Docker Desktop, weftspun-hailo-dfc:latest):
#
#     warm caches   406s, whole suite, 7/7
#     cold caches   >600s, and the cross-check test was killed mid-optimize
#
# The cold number is the one that matters, because it is what a fresh clone does. 600_000 was
# adequate for the eight cases that existed before chain1/2/4/8 were added and was never
# raised to match them, so the suite came to ask for more optimize passes than its own clock
# allowed. A budget that only holds once the caches are warm fails the first person to run it.
ExUnit.start(timeout: 1_800_000)
