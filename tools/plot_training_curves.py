"""
Plot 3D AP (moderate, R40 @ IoU 0.7) training curves from two log files.

Usage:
  python tools/plot_training_curves.py
"""

import os
import re

import matplotlib.pyplot as plt

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)

RUNS = {
    "Baseline ": os.path.join(LOG_DIR, "baseline_100.20260418_160539"),
    "Grad. Accum.": os.path.join(LOG_DIR, "gradient_accum_150.20260420_114543"),
}

MAX_EPOCHS = 100


def parse_log(path):
    """Return list of (epoch, moderate_3d_ap_r40) parsed from a log file."""
    results = []
    epoch = None
    after_r40 = False
    r40_bbox_seen = False

    with open(path) as f:
        for line in f:
            line = line.strip()

            m = re.search(r"Test Epoch (\d+)", line)
            if m:
                epoch = int(m.group(1))
                after_r40 = False
                r40_bbox_seen = False
                continue

            if "Car AP_R40@0.70, 0.70, 0.70" in line:
                after_r40 = True
                r40_bbox_seen = False
                continue

            if after_r40 and line.startswith("bbox AP:"):
                r40_bbox_seen = True
                continue

            if after_r40 and r40_bbox_seen and line.startswith("bev  AP:"):
                continue

            if after_r40 and r40_bbox_seen and line.startswith("3d   AP:"):
                vals = re.findall(r"[\d.]+", line)
                if len(vals) >= 2 and epoch is not None:
                    moderate = float(vals[1])
                    results.append((epoch, moderate))
                after_r40 = False
                r40_bbox_seen = False

    return results


fig, ax = plt.subplots(figsize=(9, 5))

colors = ["#0072B2", "#E69F00"]
for (label, path), color in zip(RUNS.items(), colors):
    data = parse_log(path)
    data = [(ep, ap) for ep, ap in data if ep <= MAX_EPOCHS]
    if not data:
        print(f"WARNING: no data parsed from {path}")
        continue
    epochs, aps = zip(*sorted(data))
    ax.plot(epochs, aps, label=label, color=color, linewidth=1.8)
    # mark best
    best_idx = max(range(len(aps)), key=lambda i: aps[i])

ax.set_xlabel("Epoch")
ax.set_ylabel("3D AP moderate (R40, IoU 0.7)")
ax.set_xlim(1, MAX_EPOCHS)
ax.legend()
ax.grid(True, alpha=0.3)

out = os.path.join(LOG_DIR, "training_curves.png")
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
