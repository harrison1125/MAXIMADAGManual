"""
XRF Vanadium Atomic Percent Heatmap
=====================================
Reads VAtomicPercent.csv from each dataset's raw/ subfolder,
merges split datasets (folder 1 priority), and plots 11x11 heatmaps.

CONFIGURATION - edit this section to match your data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

plt.style.use("/Users/hpark108/Downloads/weihs lab python.mplstyle")

# ── Colormap ───────────────────────────────────────────────────────────────────
def hot_black_oob():
    """'hot' colormap with black for out-of-range and masked (blacked-out) values."""
    cmap = plt.get_cmap("hot").copy()
    cmap.set_under("black")
    cmap.set_over("black")
    cmap.set_bad("black")
    return cmap

HOT_BLACK = hot_black_oob()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/hpark108/Desktop/open-face for XRF")

# Each dataset is a list of (folder, scan_points) tuples.
# scan_points is the list of [x, y] coords from the instruction JSON for that folder.
# Folder 1 has priority: if two folders cover the same coordinate, folder 1 wins.

C_FOLDER1_PTS = [[-14.0,-20.0],[-14.0,-16.0],[-14.0,-12.0],[-14.0,-8.0],[-14.0,-4.0],[-14.0,0.0],[-14.0,4.0],[-14.0,8.0],[-14.0,12.0],[-14.0,16.0],[-14.0,20.0],[-10.0,-20.0],[-10.0,-16.0],[-10.0,-12.0],[-10.0,-8.0],[-10.0,-4.0],[-10.0,0.0],[-10.0,4.0],[-10.0,8.0],[-10.0,12.0],[-10.0,16.0],[-10.0,20.0],[-6.0,-20.0],[-6.0,-16.0],[-6.0,-12.0],[-6.0,-8.0],[-6.0,-4.0],[-6.0,0.0],[-6.0,4.0],[-6.0,8.0],[-6.0,12.0],[-6.0,16.0],[-6.0,20.0],[-2.0,-20.0],[-2.0,-16.0],[-2.0,-12.0],[-2.0,-8.0],[-2.0,-4.0],[-2.0,0.0],[-2.0,4.0],[-2.0,8.0]]
C_FOLDER2_PTS = [[-2.0,-12.0],[-2.0,-8.0],[-2.0,-4.0],[-2.0,0.0],[-2.0,4.0],[-2.0,8.0],[-2.0,12.0],[-2.0,16.0],[-2.0,20.0],[2.0,-20.0],[2.0,-16.0],[2.0,-12.0],[2.0,-8.0],[2.0,-4.0],[2.0,0.0],[2.0,4.0],[2.0,8.0],[2.0,12.0],[2.0,16.0],[2.0,20.0],[6.0,-20.0],[6.0,-16.0],[6.0,-12.0],[6.0,-8.0],[6.0,-4.0],[6.0,0.0],[6.0,4.0],[6.0,8.0],[6.0,12.0],[6.0,16.0],[6.0,20.0],[10.0,-20.0],[10.0,-16.0],[10.0,-12.0],[10.0,-8.0],[10.0,-4.0],[10.0,0.0],[10.0,4.0],[10.0,8.0],[10.0,12.0],[10.0,16.0],[10.0,20.0],[14.0,-20.0],[14.0,-16.0],[14.0,-12.0],[14.0,-8.0],[14.0,-4.0],[14.0,0.0],[14.0,4.0],[14.0,8.0],[14.0,12.0],[14.0,16.0],[14.0,20.0],[18.0,-20.0],[18.0,-16.0],[18.0,-12.0],[18.0,-8.0],[18.0,-4.0],[18.0,0.0],[18.0,4.0],[18.0,8.0],[18.0,12.0],[18.0,16.0],[18.0,20.0],[22.0,-20.0],[22.0,-16.0],[22.0,-12.0],[22.0,-8.0],[22.0,-4.0],[22.0,0.0],[22.0,4.0],[22.0,8.0],[22.0,12.0],[22.0,16.0],[22.0,20.0],[26.0,-20.0],[26.0,-16.0],[26.0,-12.0],[26.0,-8.0],[26.0,-4.0],[26.0,0.0],[26.0,4.0],[26.0,8.0],[26.0,12.0],[26.0,16.0],[26.0,20.0]]

D_FOLDER1_PTS = [[-14.0,-20.0],[-14.0,-16.0],[-14.0,-12.0],[-14.0,-8.0],[-14.0,-4.0],[-14.0,0.0],[-14.0,4.0],[-14.0,8.0],[-14.0,12.0],[-14.0,16.0],[-14.0,20.0],[-10.0,-20.0],[-10.0,-16.0],[-10.0,-12.0],[-10.0,-8.0],[-10.0,-4.0],[-10.0,0.0],[-10.0,4.0],[-10.0,8.0],[-10.0,12.0],[-10.0,16.0],[-10.0,20.0],[-6.0,-20.0],[-6.0,-16.0],[-6.0,-12.0],[-6.0,-8.0],[-6.0,-4.0],[-6.0,0.0],[-6.0,4.0],[-6.0,8.0],[-6.0,12.0],[-6.0,16.0],[-6.0,20.0],[-2.0,-20.0],[-2.0,-16.0],[-2.0,-12.0],[-2.0,-8.0],[-2.0,-4.0],[-2.0,0.0],[-2.0,4.0],[-2.0,8.0],[-2.0,12.0],[-2.0,16.0],[-2.0,20.0],[2.0,-20.0],[2.0,-16.0],[2.0,-12.0],[2.0,-8.0],[2.0,-4.0],[2.0,0.0],[2.0,4.0],[2.0,8.0],[2.0,12.0],[2.0,16.0],[2.0,20.0],[6.0,-20.0],[6.0,-16.0],[6.0,-12.0]]
D_FOLDER2_PTS = [[6.0,-8.0],[6.0,-4.0],[6.0,0.0],[6.0,4.0],[6.0,8.0],[6.0,12.0],[6.0,16.0],[6.0,20.0],[10.0,-20.0],[10.0,-16.0],[10.0,-12.0],[10.0,-8.0],[10.0,-4.0],[10.0,0.0],[10.0,4.0],[10.0,8.0],[10.0,12.0],[10.0,16.0],[10.0,20.0],[14.0,-20.0],[14.0,-16.0],[14.0,-12.0],[14.0,-8.0],[14.0,-4.0],[14.0,0.0],[14.0,4.0],[14.0,8.0],[14.0,12.0],[14.0,16.0],[14.0,20.0],[18.0,-20.0],[18.0,-16.0],[18.0,-12.0],[18.0,-8.0],[18.0,-4.0],[18.0,0.0],[18.0,4.0],[18.0,8.0],[18.0,12.0],[18.0,16.0],[18.0,20.0],[22.0,-20.0],[22.0,-16.0],[22.0,-12.0],[22.0,-8.0],[22.0,-4.0],[22.0,0.0],[22.0,4.0],[22.0,8.0],[22.0,12.0],[22.0,16.0],[22.0,20.0],[26.0,-20.0],[26.0,-16.0],[26.0,-12.0],[26.0,-8.0],[26.0,-4.0],[26.0,0.0],[26.0,4.0],[26.0,8.0],[26.0,12.0],[26.0,16.0],[26.0,20.0]]

DATASETS = {
    "A - S1R1C4 rolled": [
        (BASE / "JHBMAI00003-S1R1C4_jhmai00003-114-rolled_1_1_2026-03-07_15-46-03", None),
    ],
    "B - S1R3C3 rolled": [
        (BASE / "JHBMAI00003-S1R3C3_jhmai00003-133-rolled_1_1_2026-03-06_23-06-25", None),
    ],
    "C - S1R1C4 non-rolled": [
        (BASE / "JHXMAA00009_jhmai00003-114_1_1_2026-03-04_20-31-53", D_FOLDER1_PTS),  # 58 pts, priority
        (BASE / "JHXMAA00009_jhmai00003-114_1_1_2026-03-05_14-57-47", D_FOLDER2_PTS),  # 63 pts, fills rest
    ],
    "D - S1R3C3 non-rolled": [
        (BASE / "JHBMAI00003-S1R3C3_696174b27dd7453db11787e9_0_716_2026-03-03_17-10-15", C_FOLDER1_PTS),  # 41 pts, priority
        (BASE / "JHBMAI00003-S1R3C3_jhmai00003-133-pt2_1_1_2026-03-03_19-22-45",        C_FOLDER2_PTS),  # 86 pts, fills rest
    ],
}

TOTAL_POINTS = 121
GRID_ROWS    = 11
GRID_COLS    = 11

# Full 11x11 grid coordinates
X_COORDS = list(range(-14, 27, 4))   # [-14, -10, ..., 26] - rows
Y_COORDS = list(range(-20, 21, 4))   # [-20, -16, ...,  20] - cols
CSV_FILENAME = "VAtomicPercent.csv"
VALUE_COL    = "V Atomic Percent"
SCAN_COL     = "Scan Point"

# ── Loader ─────────────────────────────────────────────────────────────────────
def load_csv(folder):
    """Load the CSV from a folder's raw/ subfolder. Returns a DataFrame sorted by Scan Point."""
    path = folder / "raw" / CSV_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, sep=",")
    df = df.sort_values(SCAN_COL).reset_index(drop=True)
    return df


def coord_to_grid(x, y):
    """Convert [x, y] scan coordinate to (row, col) in the 11x11 grid."""
    row = X_COORDS.index(x)
    col = Y_COORDS.index(y)
    return row, col


def merge_dataset(folder_specs):
    """
    Build an 11x11 grid of V Atomic % values.
    For single-folder datasets (scan_points=None), values are placed in sequential order.
    For split datasets, each folder's scan points map to explicit grid coordinates;
    folder 1 has priority over folder 2 for any overlapping cells.
    """
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)

    if len(folder_specs) == 1 and folder_specs[0][1] is None:
        # Single complete folder - sequential order matches the standard grid layout
        folder, _ = folder_specs[0]
        df = load_csv(folder)
        values = df[VALUE_COL].values
        if len(values) != TOTAL_POINTS:
            print(f"  WARNING: expected {TOTAL_POINTS} points, got {len(values)}")
        grid = values[:TOTAL_POINTS].reshape(GRID_ROWS, GRID_COLS)
        return grid

    # Multi-folder: place values by coordinate, folder 1 first (so it wins on overlap)
    for folder, scan_points in folder_specs:
        df = load_csv(folder)
        values = df[VALUE_COL].values
        if len(values) != len(scan_points):
            print(f"  WARNING: {folder.name} has {len(values)} values "
                  f"but {len(scan_points)} scan points listed")
        for i, (x, y) in enumerate(scan_points):
            if i >= len(values):
                break
            row, col = coord_to_grid(x, y)
            if np.isnan(grid[row, col]):  # only fill empty cells (folder 1 priority)
                grid[row, col] = values[i]

    missing = np.sum(np.isnan(grid))
    if missing > 0:
        print(f"  WARNING: {missing} grid cells have no data (NaN)")

    return grid


# Shared color scale across all datasets
GLOBAL_VMIN = 0
GLOBAL_VMAX = 30
GLOBAL_NORM = mcolors.Normalize(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)

# Per-dataset black-out thresholds: values outside [black_min, black_max] render black.
# Values inside this range are plotted on the shared 10-30 scale.
BLACKOUT = {
    "A - S1R1C4 rolled":     (5, 20),
    "B - S1R3C3 rolled":     (15, 30),
    "C - S1R1C4 non-rolled": (5, 20),
    "D - S1R3C3 non-rolled": (15, 30),
}


def apply_blackout(grid, dataset_name):
    """Mask cells outside the dataset's black-out range so they render black.
    For S1R3C3 datasets, also mask the entire outermost ring of cells."""
    bmin, bmax = BLACKOUT.get(dataset_name, (GLOBAL_VMIN, GLOBAL_VMAX))
    mask = (grid < bmin) | (grid > bmax)

    if "S1R3C3" in dataset_name:
        mask[0, :]  = True  # top row
        mask[-1, :] = True  # bottom row
        mask[:, 0]  = True  # left col
        mask[:, -1] = True  # right col

    return np.ma.masked_where(mask, grid)


# ── Plotter ────────────────────────────────────────────────────────────────────
def plot_heatmap(values, dataset_name, output_dir):
    grid = apply_blackout(values, dataset_name)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        grid,
        cmap=HOT_BLACK,
        aspect="equal",
        norm=GLOBAL_NORM,
    )
    ax.set_title(f"V Atomic % - {dataset_name}", fontsize=12, pad=10)
    ax.set_xlabel("Y position (mm)")
    ax.set_ylabel("X position (mm)")
    ax.set_xticks(range(GRID_COLS))
    ax.set_xticklabels(Y_COORDS)
    ax.set_yticks(range(GRID_ROWS))
    ax.set_yticklabels(X_COORDS)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("V Atomic %")

    plt.tight_layout()
    safe_name = dataset_name.replace(" ", "_").replace("—", "-").replace("—", "-")
    out_path = output_dir / f"{safe_name}_V_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Comparison plot (all 4 datasets, shared color scale) ──────────────────────
def plot_comparison(all_values, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("V Atomic % - All Datasets", fontsize=14)

    for ax, (name, values) in zip(axes.flat, all_values.items()):
        grid = apply_blackout(values, name)
        im = ax.imshow(grid, cmap=HOT_BLACK, aspect="equal", norm=GLOBAL_NORM)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Y position (mm)")
        ax.set_ylabel("X position (mm)")
        ax.set_xticks(range(GRID_COLS))
        ax.set_xticklabels(Y_COORDS)
        ax.set_yticks(range(GRID_ROWS))
        ax.set_yticklabels(X_COORDS)

    # Single shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="V Atomic %")

    plt.tight_layout(rect=[0, 0, 0.87, 1])
    out_path = output_dir / "all_datasets_V_heatmap_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    output_dir = BASE / "xrf_outputs"
    output_dir.mkdir(exist_ok=True)

    all_values = {}

    for dataset_name, folders in DATASETS.items():
        print(f"\nProcessing: {dataset_name}")
        values = merge_dataset(folders)
        print(f"  Loaded grid shape: {values.shape}, non-NaN cells: {np.sum(~np.isnan(values))}")
        all_values[dataset_name] = values
        plot_heatmap(values, dataset_name, output_dir)

    print("\nGenerating comparison plot...")
    plot_comparison(all_values, output_dir)

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
