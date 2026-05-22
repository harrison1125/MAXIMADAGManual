import os
import fabio
import numpy as np
import matplotlib.pyplot as plt
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import Inputs


def azimuthally_integrate_files(
    input_directory: str,
    poni_file: str,
    npt: int = 10000,
    # Should be dynamic eventually
    y_limits=(0, 200),
    x_limits=(10, 80),
) -> None:
    """
    Perform azimuthal integration on all TIFF images in a directory tree.

    This function walks through `input_directory`, identifies `.tif` and `.tiff`
    files, performs 1D azimuthal integration using a single pyFAI `.poni`
    calibration file, and saves both numerical results and diagnostic plots.

    Parameters
    ----------
    input_directory : str
        Root directory containing TIFF images to be processed.
    poni_file : str
        Path to the pyFAI calibration (.poni) file.
    npt : int, optional
        Number of radial bins for integration.
    y_limits : tuple of float, optional
        Y-axis limits for output plots.
    x_limits : tuple of float, optional
        X-axis limits for output plots.

    Returns
    -------
    None
        Results are written to disk as `.dat` and `.png` files.
    """
    # Initialize azimuthal integrator once
    ai = AzimuthalIntegrator()
    ai.load(poni_file)

    results = {}

    for dirpath, _, filenames in os.walk(input_directory):
        for filename in filenames:
            base_name, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in (".tif", ".tiff"):
                continue

            input_path = os.path.join(dirpath, filename)
            output_dat = os.path.join(dirpath, f"{base_name}.dat")
            output_png = os.path.join(dirpath, f"{base_name}.png")

            try:
                image = fabio.open(input_path).data

                # pyFAI integrate1d returns: radial axis (2theta), intensity (1D)
                two_theta, intensity = ai.integrate1d(image, npt=npt)

                # Save numerical output
                np.savetxt(
                    output_dat,
                    np.column_stack((two_theta, intensity)),
                    header="2theta Intensity",
                    comments="",
                )

                # Plot integrated pattern
                plt.figure(figsize=(8, 5))
                plt.plot(two_theta, intensity, lw=2)
                plt.xlabel(r"2$\theta$ (degrees)", fontsize=15)
                plt.ylabel("Intensity", fontsize=15)
                plt.xlim(*x_limits)
                plt.ylim(*y_limits)
                plt.tick_params(axis="both", which="major", labelsize=15)
                plt.tight_layout()
                plt.savefig(output_png, dpi=150)
                plt.close()

                print(f"Saved: {output_dat}, {output_png}")

                results[base_name] = {
                    "two_theta": two_theta,
                    "intensity": intensity,
                }

            except Exception as exc:
                print(f"Failed to process {input_path}: {exc}")


if __name__ == "__main__":
    azimuthally_integrate_files(
        input_directory=Inputs.root_dir,
        poni_file=Inputs.poni_file,
    )
