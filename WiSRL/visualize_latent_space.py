import argparse
import json
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_simCLR import ComplexDataset


def load_model(module_path: str, class_name: str, kwargs: Dict[str, Any]) -> torch.nn.Module:
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")
    cls = getattr(module, class_name)
    return cls(**kwargs)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
        
# model_pre.load_state_dict(Pre_checkpoint['model_state_dict'], strict=False)

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected}")


def pick_embedding(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in ("proj", "projection", "z", "embedding", "feat", "features", "out"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]

    raise ValueError("Cannot extract embedding from model output")


def forward_views(
    model: torch.nn.Module,
    amp1: torch.Tensor,
    pha1: torch.Tensor,
    amp2: torch.Tensor,
    pha2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        try:
            out = model(amp1, pha1, amp2, pha2)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                return pick_embedding(out[0]), pick_embedding(out[1])
            if isinstance(out, dict) and "view1" in out and "view2" in out:
                return pick_embedding(out["view1"]), pick_embedding(out["view2"])
        except TypeError:
            pass

        out1 = model(amp1, pha1)
        out2 = model(amp2, pha2)
        return pick_embedding(out1), pick_embedding(out2)


def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return z1 @ z2.t()


def plot_heatmap(sim: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    ax.set_title("View1 vs View2 cosine similarity")
    ax.set_xlabel("View-2 index")
    ax.set_ylabel("View-1 index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # ax.set_xticks(range(sim.shape[1]))
    # ax.set_yticks(range(sim.shape[0]))
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_kwargs(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize latent-space similarity heatmap for two views")
    parser.add_argument("--data-folder", type=str, required=True, help="Folder containing .mat CSI samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-module", type=str, required=True, help="Python module path for the model")
    parser.add_argument("--model-class", type=str, required=True, help="Model class name")
    parser.add_argument("--model-kwargs", type=str, default="", help="JSON string for model init kwargs")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--out", type=str, default="visualization_outputs/latent_view_similarity_preprocess_type2.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = ComplexDataset(args.data_folder)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = load_model(args.model_module, args.model_class, parse_kwargs(args.model_kwargs))
    load_checkpoint(model, Path(args.ckpt))
    model.to(args.device)

    amp_v1, pha_v1, amp_v2, pha_v2, _ = next(iter(loader))
    amp_v1 = amp_v1.to(args.device)
    pha_v1 = pha_v1.to(args.device)
    amp_v2 = amp_v2.to(args.device)
    pha_v2 = pha_v2.to(args.device)

    z1, z2 = forward_views(model, amp_v1, pha_v1, amp_v2, pha_v2)
    sim = cosine_similarity_matrix(z1, z2).cpu().numpy()

    plot_heatmap(sim, Path(args.out))
    print(f"[ok] Saved heatmap to: {args.out}")


if __name__ == "__main__":
    main()


# runs:
# python3 visualize_latent_space.py   --data-folder '/mnt/data/keran/project/WiSRL/dataset/WIDAR_Pre'   --model-module models.pretraining_Biblock_model_simCLR   --model-class WiSRL_pre   --model-kwargs '{"encoder_nlayers":3,"Proj_head":true,"proj_dim":64,"input_dim":270,"hidden_dim":64,"bimamba_type":"v2"}'   --ckpt "./model_weight/PRE/pre_100_Biblockv2_3+3link_test_singleBlock.pth"
# python3 visualize_latent_space.py   --data-folder '/mnt/data/keran/project/WiSRL/dataset/WIDAR_Pre'   --model-module models.pretraining_Biblock_model_simCLR   --model-class WiSRL_pre   --model-kwargs '{"encoder_nlayers":3,"Proj_head":true,"proj_dim":64,"input_dim":270,"hidden_dim":64,"bimamba_type":"v2"}'   --ckpt "./model_weight/PRE/pre_100_Biblockv2_3+3link_test_bz64.pth"
# python3 visualize_latent_space.py   --data-folder '/mnt/data/keran/project/WiSRL/dataset/WIDAR_Pre'   --model-module models.pretraining_Biblock_model_simCLR   --model-class WiSRL_pre   --model-kwargs '{"encoder_nlayers":3,"Proj_head":true,"proj_dim":64,"input_dim":270,"hidden_dim":64,"bimamba_type":"v2"}'   --ckpt "./model_weight/PRE/pre_100_Biblockv2_3+3link_test_preprocess_type2.pth"
# python3 visualize_latent_space.py   --data-folder '/mnt/data/keran/project/WiSRL/dataset/WIDAR_Pre'   --model-module models.pretraining_Biblock_model_simCLR   --model-class WiSRL_pre   --model-kwargs '{"encoder_nlayers":3,"Proj_head":false,"proj_dim":64,"input_dim":270,"hidden_dim":64,"bimamba_type":"v2"}'   --ckpt "./model_weight/PRE/pre_100_Biblockv2_3+3link_test_noProjHead.pth"
# python3 visualize_latent_space.py   --data-folder '/mnt/data/keran/project/WiSRL/dataset/WIDAR_Pre'   --model-module models.pretraining_model_simCLR   --model-class WiSRL_pre   --model-kwargs '{"encoder_nlayers":3,"proj_dim":64,"input_dim":270,"hidden_dim":64,"bimamba_type":"v2"}'   --ckpt "./model_weight/PRE/pre_100_Biblockv2_3+3link_test_singleBlock.pth"