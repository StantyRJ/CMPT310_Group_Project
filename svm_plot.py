import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import json

# --- Load data ---
with open("svm_search_results.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

def print_highest():
    # Extract accuracies
    accuracies = np.array([d["accuracy"]["accuracy"] for d in data])

    # Find the maximum accuracy
    max_acc = np.max(accuracies)

    # Get all configurations with the maximum accuracy
    best_configs = [d for d in data if np.isclose(d["accuracy"]["accuracy"], max_acc)]

    print(f"Highest accuracy: {max_acc:.4f}")
    print("Best hyperparameter configuration(s):")
    for config in best_configs:
        # Remove accuracy dictionary if you want cleaner output
        cfg_copy = config.copy()
        cfg_copy.pop("accuracy", None)
        print(cfg_copy)

print_highest()

# Select features/dimensions to plot (include additional hyperparams, remove training time)
features = ["kernel", "C", "gamma", "tol", "coef0", "degree", "accuracy"]
kernels = list(sorted(set(d["kernel"] for d in data)))

# Prepare y-values
ys = []
categories = []
for d in data:
    y = [
        kernels.index(d["kernel"]),  # encode kernel numerically for plotting
        d["C"],
        d["gamma"],
        d["tol"],
        d.get("coef0", 0.0),
        d.get("degree", 3),
        d["accuracy"]["accuracy"],
    ]
    ys.append(y)
    categories.append(kernels.index(d["kernel"]) + 1)

ys = np.array(ys, dtype=float)
categories = np.array(categories)
N, D = ys.shape

# Make C, gamma, tol logarithmic (plot log10 values). Protect against zeros.
eps = 1e-12
try:
    c_idx = features.index('C')
    gamma_idx = features.index('gamma')
    tol_idx = features.index('tol')
    ys[:, c_idx] = np.log10(np.clip(ys[:, c_idx], eps, None))
    ys[:, gamma_idx] = np.log10(np.clip(ys[:, gamma_idx], eps, None))
    ys[:, tol_idx] = np.log10(np.clip(ys[:, tol_idx], eps, None))
except ValueError:
    # If columns not found, skip
    c_idx = tol_idx = None

# Add larger jitter to reduce overlapping lines
jitter_scale = 0.03  # increased jitter
ys_jittered = ys + np.random.uniform(-jitter_scale, jitter_scale, ys.shape) * (ys.max(axis=0) - ys.min(axis=0))

# Normalize all dimensions to main axis
ymins = ys_jittered.min(axis=0)
ymaxs = ys_jittered.max(axis=0)
dys = ymaxs - ymins
ymins -= dys * 0.05
ymaxs += dys * 0.05
dys = ymaxs - ymins

zs = np.zeros_like(ys_jittered)
zs[:, 0] = ys_jittered[:, 0]
zs[:, 1:] = (ys_jittered[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

# --- Setup axes ---
fig, host = plt.subplots(figsize=(12, 6))
axes = [host] + [host.twinx() for i in range(D - 1)]
for i, ax in enumerate(axes):
    ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (D - 1)))

host.set_xlim(0, D - 1)

# Update x-axis labels to indicate log10 for C and tol
xticklabels = []
for f in features:
    if f in ['C', 'tol', 'gamma']:
        xticklabels.append(f + ' (log10)')
    else:
        xticklabels.append(f)

host.set_xticks(range(D))
host.set_xticklabels(xticklabels, fontsize=12)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)
host.xaxis.tick_top()
host.set_title('SVM Hyperparameters', fontsize=16)

# --- Plot lines colored by accuracy (colormap) ---
accs = np.array([d["accuracy"]["accuracy"] for d in data], dtype=float)
vmin, vmax = np.nanmin(accs), np.nanmax(accs)
if np.isclose(vmin, vmax):
    norm_vals = np.full_like(accs, 0.5)
else:
    norm_vals = (accs - vmin) / (vmax - vmin)

cmap = plt.cm.viridis
for j in range(N):
    line_color = cmap(norm_vals[j])
    verts = list(zip([x for x in np.linspace(0, D - 1, D * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(
        path, facecolor='none', lw=1,
        edgecolor=line_color,
        alpha=0.7
    )
    host.add_patch(patch)

# Add colorbar for accuracy
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array(accs)
cbar = fig.colorbar(sm, ax=host, fraction=0.046, pad=0.04)
cbar.set_ticks([])

# --- Kernel labels on kernel axis ---
kernel_idx = features.index('kernel')
# Set ticks at integer kernel values on that axis and label them
axis_for_kernel = axes[kernel_idx]
axis_for_kernel.set_yticks(list(range(len(kernels))))
axis_for_kernel.set_yticklabels(kernels, fontsize=10)
axis_for_kernel.set_ylabel('parameter')

plt.tight_layout()
plt.subplots_adjust(right=0.95)
plt.savefig("svm_parallel_coordinates.png", dpi=300)
plt.show()
