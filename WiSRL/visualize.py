import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat


try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for visualization. Install it with: pip install matplotlib"
    ) from exc


def list_mat_files(data_folder: str) -> List[str]:
    files = [
        os.path.join(data_folder, name)
        for name in os.listdir(data_folder)
        if name.endswith(".mat")
    ]
    files.sort()
    return files


def load_amplitude_phase(mat_path: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = loadmat(mat_path)
    if key not in data:
        keys = ", ".join(sorted(data.keys()))
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available keys: {keys}")

    complex_data = data[key]
    amplitude = np.abs(complex_data)
    phase = np.angle(complex_data)
    return amplitude, phase


def plot_amplitude_phase(
    amplitude: np.ndarray,
    phase: np.ndarray,
    antenna: Optional[int],
    out_path: Optional[str],
    show: bool,
    title_prefix: str,
) -> None:
    if amplitude.ndim == 2:
        amplitude = amplitude[:, :, None]
        phase = phase[:, :, None]

    if amplitude.ndim != 3 or phase.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays [T, L, D], got {amplitude.shape} and {phase.shape}"
        )

    num_antennas = amplitude.shape[2]
    if antenna is None:
        antenna_indices = list(range(num_antennas))
    else:
        if antenna < 0 or antenna >= num_antennas:
            raise ValueError(f"Antenna index out of range: {antenna}")
        antenna_indices = [antenna]

    cols = len(antenna_indices)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6), squeeze=False)

    for i, d in enumerate(antenna_indices):
        amp_img = amplitude[:, :, d].T
        pha_img = phase[:, :, d].T

        ax_amp = axes[0, i]
        im_amp = ax_amp.imshow(amp_img, aspect="auto", origin="lower")
        ax_amp.set_title(f"{title_prefix}Amp D={d}")
        ax_amp.set_xlabel("Time")
        ax_amp.set_ylabel("Subcarrier")
        fig.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)

        ax_pha = axes[1, i]
        im_pha = ax_pha.imshow(pha_img, aspect="auto", origin="lower", cmap="twilight")
        ax_pha.set_title(f"{title_prefix}Phase D={d}")
        ax_pha.set_xlabel("Time")
        ax_pha.set_ylabel("Subcarrier")
        fig.colorbar(im_pha, ax=ax_pha, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize amplitude and phase from .mat CSI data.")
    parser.add_argument("--data-folder", type=str, help="Folder containing .mat files")
    parser.add_argument("--mat-path", type=str, help="Path to a single .mat file")
    parser.add_argument("--index", type=int, default=0, help="Index in data folder (sorted)")
    parser.add_argument("--key", type=str, default="CSI_result", help="Key name in .mat file")
    parser.add_argument("--antenna", type=int, default=None, help="Antenna index to plot")
    parser.add_argument("--output", type=str, default=None, help="Save plot to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")

    args = parser.parse_args()

    if not args.mat_path and not args.data_folder:
        raise SystemExit("Provide --mat-path or --data-folder")

    if args.mat_path:
        mat_path = args.mat_path
    else:
        files = list_mat_files(args.data_folder)
        if not files:
            raise SystemExit(f"No .mat files found in {args.data_folder}")
        if args.index < 0 or args.index >= len(files):
            raise SystemExit(f"Index out of range: {args.index}, total files: {len(files)}")
        mat_path = files[args.index]

    amplitude, phase = load_amplitude_phase(mat_path, args.key)
    title_prefix = os.path.basename(mat_path) + " "
    plot_amplitude_phase(
        amplitude,
        phase,
        args.antenna,
        args.output,
        not args.no_show,
        title_prefix,
    )


if __name__ == "__main__":
    main()
