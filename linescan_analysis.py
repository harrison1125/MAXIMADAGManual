"""
Center Row Line Scan (x = 6 mm)
=================================
Extracts the center row (x=6, y=-20 to 20) from each dataset and plots:
  Panel 1: V Atomic % + lattice parameter a (twin x-axes, shared y)
  Panel 2: XRD heatmap (Q vs y position, log-normalized intensity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.style.use("/Users/hpark108/Downloads/weihs lab python.mplstyle")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/hpark108/Desktop/open-face for XRF")

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
        (BASE / "JHXMAA00009_jhmai00003-114_1_1_2026-03-04_20-31-53", D_FOLDER1_PTS),
        (BASE / "JHXMAA00009_jhmai00003-114_1_1_2026-03-05_14-57-47", D_FOLDER2_PTS),
    ],
    "D - S1R3C3 non-rolled": [
        (BASE / "JHBMAI00003-S1R3C3_696174b27dd7453db11787e9_0_716_2026-03-03_17-10-15", C_FOLDER1_PTS),
        (BASE / "JHBMAI00003-S1R3C3_jhmai00003-133-pt2_1_1_2026-03-03_19-22-45",        C_FOLDER2_PTS),
    ],
}

# Grid definition
TOTAL_POINTS = 121
GRID_ROWS    = 11
GRID_COLS    = 11
X_COORDS     = list(range(-14, 27, 4))
Y_COORDS     = list(range(-20, 21, 4))

# Interior line scans: skip outermost index (0 and 10) on each axis
# axis='row' → fixed x (row index), vary y;  axis='col' → fixed y (col index), vary x
LINE_SCANS = (
    [(f"x={X_COORDS[i]}mm", "row", X_COORDS[i]) for i in range(1, 10)] +
    [(f"y={Y_COORDS[j]}mm", "col", Y_COORDS[j]) for j in range(1, 10)]
)

# XRD heatmap settings
Q_MIN     = 24
Q_MAX     = 63
Q_NPOINTS = 1000
EPSILON   = 1e-6

# ── Loaders ────────────────────────────────────────────────────────────────────
def load_xrf(folder):
    path = folder / "raw" / "VAtomicPercent.csv"
    if not path.exists():
        raise FileNotFoundError(f"XRF CSV not found: {path}")
    df = pd.read_csv(path, sep=",")
    df = df.sort_values("Scan Point").reset_index(drop=True)
    return df["V Atomic Percent"].values


def load_lattice(folder):
    path = folder / f"{folder.name}_lattice_parameters.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Lattice xlsx not found: {path}")
    df = pd.read_excel(path)
    df["_idx"] = df["scan_point"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("_idx").reset_index(drop=True)
    return df["a_A_avg"].values


def coord_to_grid(x, y):
    return X_COORDS.index(x), Y_COORDS.index(y)


def build_grid(folder_specs, loader_fn):
    """Build 11x11 grid using a given loader function."""
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)

    if len(folder_specs) == 1 and folder_specs[0][1] is None:
        folder, _ = folder_specs[0]
        values = loader_fn(folder)
        if len(values) != TOTAL_POINTS:
            print(f"  WARNING: expected {TOTAL_POINTS}, got {len(values)}")
        return values[:TOTAL_POINTS].reshape(GRID_ROWS, GRID_COLS)

    for folder, scan_points in folder_specs:
        values = loader_fn(folder)
        for i, (x, y) in enumerate(scan_points):
            if i >= len(values):
                break
            row, col = coord_to_grid(x, y)
            if np.isnan(grid[row, col]):
                grid[row, col] = values[i]

    return grid


def load_dat(path):
    """Load a .dat file (Q, intensity) skipping any header lines."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                data.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # skip header
    arr = np.array(data)
    return arr[:, 0], arr[:, 1]


def build_xrd_heatmap(folder_specs, axis, fixed_idx):
    """
    Build a (11 × Q_NPOINTS) heatmap for a line scan.
    axis='row': fixed row (fixed_idx), iterate over cols (varying y)
    axis='col': fixed col (fixed_idx), iterate over rows (varying x)
    """
    q_grid = np.linspace(Q_MIN, Q_MAX, Q_NPOINTS)

    # Build cell_source map: (row, col) → (folder, local_index)
    cell_source = {}
    if len(folder_specs) == 1 and folder_specs[0][1] is None:
        folder, _ = folder_specs[0]
        for i in range(TOTAL_POINTS):
            row = i // GRID_COLS
            col = i % GRID_COLS
            cell_source[(row, col)] = (folder, i)
    else:
        for folder, scan_points in folder_specs:
            for i, (x, y) in enumerate(scan_points):
                row, col = coord_to_grid(x, y)
                if (row, col) not in cell_source:
                    cell_source[(row, col)] = (folder, i)

    # Collect the 11 scan points along the line
    n_steps = GRID_ROWS if axis == 'col' else GRID_COLS
    raw_log = []
    for step in range(n_steps):
        cell = (fixed_idx, step) if axis == 'row' else (step, fixed_idx)
        if cell not in cell_source:
            raw_log.append(None)
            continue
        folder, local_idx = cell_source[cell]
        dat_path = folder / f"scan_point_{local_idx}.dat"
        if not dat_path.exists():
            print(f"  WARNING: {dat_path} not found")
            raw_log.append(None)
            continue
        try:
            q, intensity = load_dat(dat_path)
            raw_log.append((q, np.log10(intensity + EPSILON)))
        except Exception as e:
            print(f"  ERROR reading {dat_path}: {e}")
            raw_log.append(None)

    # Shared normalization: median of per-spectrum mins, global max
    per_mins, global_max = [], -np.inf
    for entry in raw_log:
        if entry is None:
            continue
        q, log_i = entry
        mask = (q >= Q_MIN) & (q <= Q_MAX)
        if mask.sum() == 0:
            continue
        per_mins.append(np.min(log_i[mask]))
        global_max = max(global_max, np.max(log_i))

    shared_vmin = np.median(per_mins) if per_mins else 0.0
    shared_vmax = global_max if global_max > shared_vmin else shared_vmin + 1.0

    # Normalize and interpolate
    rows_list = []
    for entry in raw_log:
        if entry is None:
            rows_list.append(np.zeros(Q_NPOINTS))
            continue
        q, log_i = entry
        log_i_norm = (log_i - shared_vmin) / (shared_vmax - shared_vmin + EPSILON)
        rows_list.append(np.interp(q_grid, q, log_i_norm, left=0, right=0))

    return np.array(rows_list), q_grid


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_combined(dataset_name, xrf_line, lat_line, xrd_heatmap, q_grid,
                  positions, pos_label, scan_label, output_dir):
    """
    Two-panel figure:
      Left:  V Atomic % + lattice parameter a (twin x-axes)
      Right: XRD heatmap (Q on x, position on y, shared y-axis)
    """
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.05)

    pos    = np.array(positions)
    p_half = (pos[1] - pos[0]) / 2

    # ── Panel 1: XRF + lattice ─────────────────────────────────────────────────
    ax_xrf = fig.add_subplot(gs[0])
    ax_xrf.plot(xrf_line, pos, marker='o', linestyle='-',
                color='black', label='V (at%)')
    ax_xrf.set_xlabel("V Atomic %")
    ax_xrf.set_ylabel(f"{pos_label} position (mm)")
    ax_xrf.set_yticks(pos)
    ax_xrf.set_ylim(pos[0] - p_half, pos[-1] + p_half)

    ax_lat = ax_xrf.twiny()
    ax_lat.plot(lat_line, pos, marker='s', linestyle='-',
                color='#56B4E9', label=r'$a$ ($\AA$)')
    ax_lat.set_xlabel(r"$a$ ($\AA$)")

    lines1, labels1 = ax_xrf.get_legend_handles_labels()
    lines2, labels2 = ax_lat.get_legend_handles_labels()
    ax_xrf.legend(lines1 + lines2, labels1 + labels2, loc='best',
                  fontsize=9, frameon=False)

    # ── Panel 2: XRD heatmap ───────────────────────────────────────────────────
    ax_xrd = fig.add_subplot(gs[1], sharey=ax_xrf)

    im = ax_xrd.imshow(
        xrd_heatmap,
        aspect='auto',
        extent=[q_grid[0], q_grid[-1], pos[0] - p_half, pos[-1] + p_half],
        origin='lower',
        cmap='cividis',
        vmin=0, vmax=1,
    )
    ax_xrd.set_xlabel(r"$Q$ (nm$^{-1}$)")
    ax_xrd.set_xlim(Q_MIN, Q_MAX)
    plt.setp(ax_xrd.get_yticklabels(), visible=False)

    cbar = plt.colorbar(im, ax=ax_xrd, pad=0.02, fraction=0.046)
    cbar.set_label("Log norm. intensity")

    fig.suptitle(f"{scan_label} line scan  |  {dataset_name}", fontsize=11)

    plt.tight_layout()
    safe_name = dataset_name.replace(" ", "_").replace("/", "-")
    out_path = output_dir / f"{safe_name}_{scan_label}_linescan.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    output_dir = BASE / "linescan_outputs"
    output_dir.mkdir(exist_ok=True)

    for scan_label, axis, fixed_val in LINE_SCANS:
        fixed_idx  = X_COORDS.index(fixed_val) if axis == 'row' else Y_COORDS.index(fixed_val)
        positions  = (Y_COORDS if axis == 'row' else X_COORDS)[1:10]  # trim edges
        pos_label  = "Y" if axis == 'row' else "X"
        print(f"\n=== Line scan: {scan_label} ===")

        for dataset_name, folder_specs in DATASETS.items():
            print(f"\n  Processing: {dataset_name}")

            xrf_grid = build_grid(folder_specs, load_xrf)
            lat_grid = build_grid(folder_specs, load_lattice)
            xrd_heatmap, q_grid = build_xrd_heatmap(folder_specs, axis, fixed_idx)

            # Extract line and trim to interior points 1-9
            xrf_line = (xrf_grid[fixed_idx, :] if axis == 'row' else xrf_grid[:, fixed_idx])[1:10]
            lat_line = (lat_grid[fixed_idx, :] if axis == 'row' else lat_grid[:, fixed_idx])[1:10]
            xrd_heatmap = xrd_heatmap[1:10, :]  # trim first and last scan point rows

            plot_combined(dataset_name, xrf_line, lat_line, xrd_heatmap, q_grid,
                          positions, pos_label, scan_label, output_dir)

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()

