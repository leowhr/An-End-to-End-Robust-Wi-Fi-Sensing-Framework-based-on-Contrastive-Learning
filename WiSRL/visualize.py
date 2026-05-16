import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def load_complex_csi(mat_path: Path, key: str | None = None) -> np.ndarray:
    data = loadmat(str(mat_path))

    if key is not None:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {mat_path.name}. Available keys: {list(data.keys())}")
        arr = data[key]
    else:
        candidates = []
        for k, v in data.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                candidates.append((k, v))
        if not candidates:
            raise ValueError(f"No ndarray candidate found in {mat_path}")

        complex_candidates = [(k, v) for k, v in candidates if np.iscomplexobj(v)]
        picked = complex_candidates if complex_candidates else candidates
        key, arr = max(picked, key=lambda item: item[1].size)
        print(f"[info] Auto-selected key: {key}, shape={arr.shape}, complex={np.iscomplexobj(arr)}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D CSI array [T, L, D], got shape={arr.shape}")

    return arr


def normalize_amp_phase(complex_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.abs(complex_data)
    phase = np.angle(complex_data)

    amp_min = amplitude.min(axis=(0, 1), keepdims=True)
    amp_max = amplitude.max(axis=(0, 1), keepdims=True)
    amplitude = (amplitude - amp_min) / (amp_max - amp_min + 1e-8)
    phase = (phase + np.pi) / (2 * np.pi)
    return amplitude, phase


def split_antennas(
    amplitude: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, d = amplitude.shape
    if d < 2:
        raise ValueError(f"Antenna dimension must be >= 2, got D={d}")

    antennas = np.random.permutation(d)
    v1_idx = antennas[: d // 2]
    v2_idx = antennas[d // 2 :]

    amp_v1 = amplitude[:, :, v1_idx]
    amp_v2 = amplitude[:, :, v2_idx]
    phase_v1 = phase[:, :, v1_idx]
    phase_v2 = phase[:, :, v2_idx]
    return amp_v1, phase_v1, amp_v2, phase_v2, v1_idx, v2_idx


def interp_resize_along_time(x: np.ndarray, target_len: int) -> np.ndarray:
    old_len = x.shape[0]
    if old_len == target_len:
        return x.copy()

    old_idx = np.arange(old_len)
    new_idx = np.linspace(0, old_len - 1, target_len)

    flat = x.reshape(old_len, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=x.dtype)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(new_idx, old_idx, flat[:, j])
    return out.reshape((target_len,) + x.shape[1:])


def random_crop_with_meta(
    x: np.ndarray,
    crop_ratio: float,
) -> tuple[np.ndarray, int, int]:
    seq_len = x.shape[0]
    crop_len = int(seq_len * crop_ratio)
    crop_len = min(crop_len, seq_len)

    if crop_len >= seq_len:
        return x.copy(), 0, seq_len

    start_idx = int(np.random.randint(0, seq_len - crop_len + 1))
    end_idx = start_idx + crop_len
    crop_x = x[start_idx:end_idx]
    resized = interp_resize_along_time(crop_x, seq_len)
    return resized, start_idx, end_idx


def random_mask_with_meta(
    x: np.ndarray,
    mask_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    seq_len = x.shape[0]
    mask_len = int(seq_len * mask_ratio)
    mask_len = min(mask_len, seq_len)

    masked_idx = np.array([], dtype=int)
    if mask_len > 0:
        masked_idx = np.random.choice(seq_len, size=mask_len, replace=False)
        masked_idx = np.sort(masked_idx)

    out = x.copy()
    out[masked_idx] = 0
    return out, masked_idx


def apply_type1_augment(
    x: np.ndarray,
    crop_ratio: float,
    circular_shift: int,
    mask_ratio: float,
) -> dict:
    crop_out, crop_start, crop_end = random_crop_with_meta(x, crop_ratio)
    shift_out = np.roll(crop_out, shift=circular_shift, axis=0)
    final_out, masked_idx = random_mask_with_meta(shift_out, mask_ratio)

    return {
        "input": x,
        "crop": crop_out,
        "shift": shift_out,
        "final": final_out,
        "crop_start": crop_start,
        "crop_end": crop_end,
        "shift_step": circular_shift,
        "masked_idx": masked_idx,
        "crop_ratio": crop_ratio,
        "mask_ratio": mask_ratio,
    }


def flatten_feature_map(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def plot_heatmap(ax, arr2d: np.ndarray, title: str):
    im = ax.imshow(arr2d.T, aspect="auto", origin="lower", cmap="turbo")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Subcarrier x antenna")
    return im


def make_visualization(
    amp_raw: np.ndarray,
    amp_v1: np.ndarray,
    amp_v2: np.ndarray,
    aug_v1: dict,
    aug_v2: dict,
    out_png: Path,
):
    raw_flat = flatten_feature_map(amp_raw)
    v1_flat = flatten_feature_map(amp_v1)
    v2_flat = flatten_feature_map(amp_v2)
    v1_crop_flat = flatten_feature_map(aug_v1["crop"])
    v1_final_flat = flatten_feature_map(aug_v1["final"])
    v2_final_flat = flatten_feature_map(aug_v2["final"])

    fig = plt.figure(figsize=(18, 11), dpi=150)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.9], hspace=0.35, wspace=0.22)

    ax00 = fig.add_subplot(gs[0, 0])
    im00 = plot_heatmap(ax00, raw_flat, "Before split: normalized amplitude")
    fig.colorbar(im00, ax=ax00, fraction=0.046)

    ax01 = fig.add_subplot(gs[0, 1])
    im01 = plot_heatmap(ax01, v1_flat, "After antenna split: View-1")
    fig.colorbar(im01, ax=ax01, fraction=0.046)

    ax02 = fig.add_subplot(gs[0, 2])
    im02 = plot_heatmap(ax02, v2_flat, "After antenna split: View-2")
    fig.colorbar(im02, ax=ax02, fraction=0.046)

    ax10 = fig.add_subplot(gs[1, 0])
    im10 = plot_heatmap(ax10, v1_crop_flat, "View-1 after random crop + resize")
    fig.colorbar(im10, ax=ax10, fraction=0.046)

    ax11 = fig.add_subplot(gs[1, 1])
    im11 = plot_heatmap(ax11, v1_final_flat, "View-1 final")
    fig.colorbar(im11, ax=ax11, fraction=0.046)

    ax12 = fig.add_subplot(gs[1, 2])
    im12 = plot_heatmap(ax12, v2_final_flat, "View-2 final")
    fig.colorbar(im12, ax=ax12, fraction=0.046)

    ch = v1_flat.shape[1] // 2
    t = np.arange(v1_flat.shape[0])
    line_before = v1_flat[:, ch]
    line_after = v1_final_flat[:, ch]

    ax20 = fig.add_subplot(gs[2, :2])
    ax20.plot(t, line_before, lw=1.2, label="before augmentation")
    ax20.plot(t, line_after, lw=1.2, label="after augmentation")
    ax20.axvspan(
        aug_v1["crop_start"],
        max(aug_v1["crop_end"] - 1, aug_v1["crop_start"] + 1),
        alpha=0.18,
        color="tab:orange",
        label="crop region on original timeline",
    )
    if aug_v1["masked_idx"].size > 0:
        ax20.scatter(
            aug_v1["masked_idx"],
            line_after[aug_v1["masked_idx"]],
            s=8,
            color="tab:red",
            alpha=0.65,
            label="masked time steps",
        )
    ax20.set_title("Physical trace (one subcarrier-antenna channel)")
    ax20.set_xlabel("Time index")
    ax20.set_ylabel("Normalized amplitude")
    ax20.legend(loc="upper right", ncol=2, fontsize=9)
    ax20.grid(alpha=0.2)

    ax21 = fig.add_subplot(gs[2, 2])
    ax21.axis("off")
    summary = [
        "Type1 augmentation params",
        f"View-1 crop_ratio = {aug_v1['crop_ratio']:.3f}",
        f"View-1 circular_shift = {aug_v1['shift_step']}",
        f"View-1 mask_ratio = {aug_v1['mask_ratio']:.3f}",
        "",
        f"View-2 crop_ratio = {aug_v2['crop_ratio']:.3f}",
        f"View-2 circular_shift = {aug_v2['shift_step']}",
        f"View-2 mask_ratio = {aug_v2['mask_ratio']:.3f}",
        "",
        "What this figure shows",
        "1) Antenna split creates two correlated views",
        "2) Crop+resize warps local temporal dynamics",
        "3) Circular roll shifts the entire timeline",
        "4) Random mask removes sparse time indices",
    ]
    ax21.text(0.03, 0.97, "\n".join(summary), va="top", fontsize=10, family="monospace")

    fig.suptitle("Wi-Fi CSI SimCLR Augmentation Visualization", fontsize=14, y=0.995)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def pick_sample(data_folder: Path) -> Path:
    mats = sorted(data_folder.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat file found in {data_folder}")
    return mats[0]


def main():
    parser = argparse.ArgumentParser(description="Visualize CSI antenna split and type1 augmentations")
    parser.add_argument("--mat-file", type=str, default="/mnt/data/keran/project/WiSRL/dataset/WIDAR_Pre/user3-11-4-2-3-6-link-room2.mat", help="Path to one .mat CSI sample")
    parser.add_argument("--data-folder", type=str, default=None, help="Fallback folder to auto-pick first .mat")
    parser.add_argument("--key", type=str, default="CSI_result", help="Key in .mat file; use '' for auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible views")
    parser.add_argument("--crop-min", type=float, default=0.9)
    parser.add_argument("--crop-max", type=float, default=1.0)
    parser.add_argument("--shift-min", type=int, default=-50)
    parser.add_argument("--shift-max", type=int, default=50)
    parser.add_argument("--mask-min", type=float, default=0.0)
    parser.add_argument("--mask-max", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="visualization_outputs/simclr_type1_augment.png")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.mat_file:
        mat_path = Path(args.mat_file)
    else:
        if args.data_folder is None:
            default_folder = Path(__file__).resolve().parent / ".." / "CSI_PreProcess" / "CSI_pro"
            data_folder = default_folder.resolve()
        else:
            data_folder = Path(args.data_folder)
        mat_path = pick_sample(data_folder)

    key = None if args.key == "" else args.key
    complex_data = load_complex_csi(mat_path, key=key)

    amplitude, phase = normalize_amp_phase(complex_data)
    amp_v1, phase_v1, amp_v2, phase_v2, v1_idx, v2_idx = split_antennas(amplitude, phase)

    crop_ratio1 = float(np.random.uniform(args.crop_min, args.crop_max))
    crop_ratio2 = float(np.random.uniform(args.crop_min, args.crop_max))
    shift1 = int(np.random.randint(args.shift_min, args.shift_max))
    shift2 = int(np.random.randint(args.shift_min, args.shift_max))
    mask_ratio1 = float(np.random.uniform(args.mask_min, args.mask_max))
    mask_ratio2 = float(np.random.uniform(args.mask_min, args.mask_max))

    aug_amp_v1 = apply_type1_augment(amp_v1, crop_ratio1, shift1, mask_ratio1)
    _ = apply_type1_augment(phase_v1, crop_ratio1, shift1, mask_ratio1)
    aug_amp_v2 = apply_type1_augment(amp_v2, crop_ratio2, shift2, mask_ratio2)
    _ = apply_type1_augment(phase_v2, crop_ratio2, shift2, mask_ratio2)

    out_png = Path(args.out)
    make_visualization(amplitude, amp_v1, amp_v2, aug_amp_v1, aug_amp_v2, out_png)

    print(f"[ok] Input sample: {mat_path}")
    print(f"[ok] CSI shape: {complex_data.shape}")
    print(f"[ok] Split indices view1={v1_idx.tolist()}, view2={v2_idx.tolist()}")
    print(f"[ok] Figure saved to: {out_png.resolve()}")


if __name__ == "__main__":
    main()
