"""
figure_schematics.py
--------------------
Generates a smooth random 2D landscape, clips negative values to zero,
and saves a filled contour map.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Hard-code output path here
# ---------------------------------------------------------------------------
OUTPUT_PATH = "figures/landscape_contour.pdf"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED        = 42
GRID_SIZE   = 512      # resolution of the underlying random field
SMOOTHNESS  = 40       # gaussian_filter sigma — larger = smoother
N_CONTOURS  = 20       # number of contour levels in the plot
ZERO_THRESHOLD = 0.5  # clip values below this threshold

# ---------------------------------------------------------------------------
# Generate smooth random landscape
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

# Start from white noise, then smooth heavily
raw = rng.standard_normal((GRID_SIZE, GRID_SIZE))
smooth = gaussian_filter(raw, sigma=SMOOTHNESS)

# Normalise to roughly [-1, 1] so positive/negative regions are balanced
smooth /= smooth.std()

# Clip negatives to zero
landscape = np.maximum(smooth, ZERO_THRESHOLD)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
x = np.linspace(0, 1, GRID_SIZE)
y = np.linspace(0, 1, GRID_SIZE)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(6, 5))

cf = ax.contourf(X, Y, landscape, levels=N_CONTOURS, cmap="viridis")
cs = ax.contour(X, Y, landscape, levels=N_CONTOURS, colors="white", linewidths=0.4, alpha=0.4)

cbar = fig.colorbar(cf, ax=ax, label="f(x, y)  [negative → 0]")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Smooth random landscape  (max(f, 0))")
ax.set_aspect("equal")

plt.tight_layout()

import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close(fig)
print(f"Saved → {OUTPUT_PATH}")
