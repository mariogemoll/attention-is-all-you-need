import matplotlib.pyplot as plt
import numpy as np

from lr_schedules import aiayn_lr, cosine_lr

TOTAL_STEPS = 100_000
WARMUP_STEPS = 4_000
BASE_LR = 7e-4

steps = np.arange(0, TOTAL_STEPS, 1_000)
lrs_aiayn = [aiayn_lr(int(step) + 1, 512, WARMUP_STEPS) for step in steps]  # step is 0-based
cosine_schedule = cosine_lr(TOTAL_STEPS, WARMUP_STEPS)
lrs_cosine = [cosine_schedule(int(step)) * BASE_LR for step in steps]

plt.figure(figsize=(10, 6))
plt.plot(steps, lrs_aiayn, label="AIAYN Learning Rate Schedule")
plt.plot(steps, lrs_cosine, label="Cosine Warmup Learning Rate Schedule")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.show()  # type: ignore[no-untyped-call]
