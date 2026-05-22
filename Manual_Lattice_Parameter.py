"""
Lattice parameter extraction from azimuthal-integration outputs.

Reads root_dir and poni_file from Inputs.py, then walks the directory tree
to find all .dat files produced by azimuthal_integrator.py.  For each
sub-directory that contains .dat files, it:

  1. Converts 2θ → Q (nm⁻¹) using the wavelength from the .poni file.
  2. Extracts peak Q positions within predefined ranges.
  3. Computes d-spacings.
  4. Uses zero-intercept linear regression to estimate the average cubic
     lattice parameter.
  5. Saves per-directory Excel files and a diagnostic plot.

Usage
-----
    python lattice_parameter.py
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import Inputs


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Q ranges in nm⁻¹  (same units as the original script)
Q_RANGES: List[Tuple[float, float]] = [
    (29.0, 32.0),
    (47.0, 51.0)
]

# Corresponding Miller indices (h, k, l)
HKL_RANGES: List[Tuple[int, int, int]] = [
    (1, 1, 1),
    (2, 2, 0)
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_wavelength_from_poni(poni_file: str) -> float:
    """
    Read the X-ray wavelength from a pyFAI .poni calibration file.

    Parameters
    ----------
    poni_file : str
        Path to the .poni file.

    Returns
    -------
    float
        Wavelength in metres.

    Raises
    ------
    ValueError
        If no ``Wavelength:`` line is found in the file.
    """
    with open(poni_file, "r") as f:
        for line in f:
            if line.strip().startswith("Wavelength:"):
                return float(line.split(":", 1)[1].strip())
    raise ValueError(f"'Wavelength:' field not found in {poni_file}")


def two_theta_to_q(two_theta_deg: np.ndarray, wavelength_m: float) -> np.ndarray:
    """
    Convert 2θ (degrees) to Q (nm⁻¹).

    Parameters
    ----------
    two_theta_deg : np.ndarray
        2θ values in degrees, as written by azimuthal_integrator.py.
    wavelength_m : float
        X-ray wavelength in metres (from the .poni file).

    Returns
    -------
    np.ndarray
        Q values in nm⁻¹.
    """
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    return (4.0 * np.pi * np.sin(theta_rad)) / (wavelength_m * 1e9)


def find_peak_q(
    q: np.ndarray,
    intensity: np.ndarray,
    q_min: float,
    q_max: float,
) -> float:
    """
    Find Q position of maximum intensity within a given Q range.

    Parameters
    ----------
    q : np.ndarray
        Q values in nm⁻¹.
    intensity : np.ndarray
        Intensity values.
    q_min : float
        Lower bound of Q range.
    q_max : float
        Upper bound of Q range.

    Returns
    -------
    float
        Q position of peak maximum.  NaN if the range contains no points.
    """
    mask = (q >= min(q_min, q_max)) & (q <= max(q_min, q_max))
    if not np.any(mask):
        return np.nan

    idx = np.argmax(intensity[mask])
    return q[mask][idx]


def collect_dat_files(root_dir: str) -> Dict[str, List[str]]:
    """
    Walk root_dir and group .dat files by their parent directory.

    Parameters
    ----------
    root_dir : str
        Root directory to search.

    Returns
    -------
    dict
        Mapping of ``{dirpath: [sorted list of .dat file paths]}``.
        Directories with no .dat files are omitted.
    """
    grouped: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root_dir):
        dat_files = sorted(
            os.path.join(dirpath, f)
            for f in filenames
            if f.lower().endswith(".dat")
        )
        if dat_files:
            grouped[dirpath] = dat_files
    return grouped


# =============================================================================
# CORE PROCESSING
# =============================================================================

def process_directory(
    dirpath: str,
    dat_files: List[str],
    wavelength_m: float,
) -> None:
    """
    Compute average cubic lattice parameters for all .dat files in a directory.

    Parameters
    ----------
    dirpath : str
        Directory containing the .dat files.
    dat_files : list of str
        Sorted list of .dat file paths to process.
    wavelength_m : float
        X-ray wavelength in metres, used for 2θ → Q conversion.

    Notes
    -----
    - Assumes cubic symmetry:  d_hkl = a / sqrt(h² + k² + l²)
    - Linear regression is forced through the origin.
    - Outputs are written to *dirpath* as an Excel file and a PNG plot.
    """
    records = []

    # -------------------------------------------------------------------------
    # Peak extraction
    # -------------------------------------------------------------------------
    for file_path in dat_files:
        try:
            data = np.loadtxt(file_path, skiprows=1, )
        except Exception as exc:
            print(f"  Could not read {file_path}: {exc}")
            continue

        two_theta = data[:, 0]
        intensity = data[:, 1]
        q_vals = two_theta_to_q(two_theta, wavelength_m)

        record: Dict = {"file": os.path.basename(file_path)}

        for (q_min, q_max), (h, k, l) in zip(Q_RANGES, HKL_RANGES):
            col_name = f"Qmax_{q_min:.1f}_{q_max:.1f}"
            record[col_name] = find_peak_q(q_vals, intensity, q_min, q_max)

        records.append(record)

    if not records:
        print(f"  No readable .dat files in {dirpath}. Skipping.")
        return

    df = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # d-spacing calculation
    # -------------------------------------------------------------------------
    inv_sqrt_hkl: List[float] = []
    d_columns: List[str] = []

    for (q_min, q_max), (h, k, l) in zip(Q_RANGES, HKL_RANGES):
        q_col = f"Qmax_{q_min:.1f}_{q_max:.1f}"
        d_col = f"d_nm_{q_min:.1f}_{q_max:.1f}"

        df[d_col] = 2.0 * np.pi / df[q_col]
        d_columns.append(d_col)
        inv_sqrt_hkl.append(1.0 / np.sqrt(h**2 + k**2 + l**2))

    # -------------------------------------------------------------------------
    # Lattice parameter regression
    # -------------------------------------------------------------------------
    a_nm = []

    for _, row in df.iterrows():
        d_vals = pd.to_numeric(row[d_columns], errors="coerce").values
        valid = ~np.isnan(d_vals)

        if np.sum(valid) < 2:
            a_nm.append(np.nan)
            continue

        X = np.array(inv_sqrt_hkl)[valid].reshape(-1, 1)
        y = d_vals[valid]

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        a_nm.append(model.coef_[0])

    df["a_nm_avg"] = a_nm
    df["a_A_avg"] = df["a_nm_avg"] * 10.0  # nm → Å

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    dir_label = os.path.basename(os.path.normpath(dirpath))
    output_xlsx = os.path.join(dirpath, f"{dir_label}_lattice_parameters.xlsx")
    output_png = os.path.join(dirpath, f"{dir_label}_lattice_parameters.png")

    df.to_excel(output_xlsx, index=False)
    print(f"  ✓ Saved {output_xlsx}")

    plt.figure(figsize=(10, 4))
    plt.plot(df["a_A_avg"], marker="o", linewidth=1.5)
    plt.xticks(range(len(df)), df["file"], rotation=90, fontsize=6)
    plt.xlabel("File")
    plt.ylabel(r"Average Lattice Parameter $a$ (Å)")
    plt.title(f"Average FCC Lattice Parameter – {dir_label}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"  ✓ Saved {output_png}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main execution loop."""
    root_dir: str = Inputs.root_dir
    poni_file: str = Inputs.poni_file

    print(f"Root directory : {root_dir}")
    print(f"PONI file      : {poni_file}")

    wavelength_m = read_wavelength_from_poni(poni_file)
    print(f"Wavelength     : {wavelength_m:.6e} m\n")

    grouped = collect_dat_files(root_dir)

    if not grouped:
        print("No .dat files found under root_dir. Exiting.")
        return

    print(f"Found .dat files in {len(grouped)} directory/directories.\n")

    for dirpath, dat_files in grouped.items():
        print(f"Processing: {dirpath}  ({len(dat_files)} file(s))")
        process_directory(dirpath, dat_files, wavelength_m)


if __name__ == "__main__":
    main()
