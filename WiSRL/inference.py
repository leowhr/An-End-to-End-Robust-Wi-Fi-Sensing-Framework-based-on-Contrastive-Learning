import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_simCLR import ComplexDataset
from models.finetune_model_simCLR import WiSRL_tune
from models.finetune_model_simCLR_single import WiSRL_tune as WiSRL_tune_single
from models.pretraining_Biblock_model_simCLR import WiSRL_pre
from models.pretrain_Biblock_model_simCLR_single import WiSRL_pre as WiSRL_pre_single
from utils import set_seed


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def count_parameters(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_params, trainable_params


def bytes_to_mib(value_in_bytes):
    return value_in_bytes / (1024 ** 2)


def build_model(input_type, input_dim_pre, hidden_dim, output_dim, bimamba_type, encoder_n_layers, device):
    if input_type == "both":
        model_pre = WiSRL_pre(
            input_dim=input_dim_pre,
            hidden_dim=hidden_dim,
            proj_dim=0,
            bimamba_type=bimamba_type,
            encoder_nlayers=encoder_n_layers,
        ).to(device)
        model_tune = WiSRL_tune(
            original_model=model_pre,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(device)
    elif input_type in {"amp", "pha"}:
        model_pre = WiSRL_pre_single(
            input_dim=input_dim_pre,
            hidden_dim=hidden_dim,
            proj_dim=0,
            bimamba_type=bimamba_type,
            encoder_nlayers=encoder_n_layers,
        ).to(device)
        model_tune = WiSRL_tune_single(
            original_model=model_pre,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(device)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    return model_tune


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def inference():
    bimamba_type = "v2"
    tune_datasize = 1.0
    process_type = "type1"
    input_type = "both"  # "both" / "amp" / "pha"
    tune_checkpoint_path = "./model_weight/Finetune/CLS/tune_100_100_Biblockv2_3+3link_test.pth"

    data_folder = "/mnt/data/keran/project/WiSRL/dataset/10fenlei(5000)"
    batch_size = 128

    seed = 624
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    input_dim_pre = 90 * 3
    hidden_dim = 64
    encoder_n_layers = 3
    output_dim = 10

    dataset = ComplexDataset(data_folder, process_type=process_type)
    dataset.set_eval(True)

    train_size = int(0.9 * len(dataset) * tune_datasize)
    val_size = len(dataset) - train_size
    _, eval_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        drop_last=False,
    )

    model = build_model(
        input_type=input_type,
        input_dim_pre=input_dim_pre,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        bimamba_type=bimamba_type,
        encoder_n_layers=encoder_n_layers,
        device=device,
    )

    load_checkpoint(model, tune_checkpoint_path, device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.3f} M)")

    criterion = torch.nn.CrossEntropyLoss()

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    baseline_allocated = 0
    baseline_reserved = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)

    with torch.inference_mode():
        for batch_idx, (amp_view1, pha_view1, amp_view2, pha_view2, label) in enumerate(eval_loader):
            amp_view1 = amp_view1.to(torch.float32).to(device)
            pha_view1 = pha_view1.to(torch.float32).to(device)
            amp_view2 = amp_view2.to(torch.float32).to(device)
            pha_view2 = pha_view2.to(torch.float32).to(device)
            label = label.to(torch.long).to(device) - 13

            if input_type == "both":
                logits = model(amp_view1, amp_view2, pha_view1, pha_view2)
            elif input_type == "amp":
                logits = model(amp_view1, amp_view2)
            else:
                logits = model(pha_view1, pha_view2)

            loss = criterion(logits, label)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            batch_size_actual = label.size(0)
            total_loss += loss.item() * batch_size_actual
            total_correct += (predictions == label).sum().item()
            total_samples += batch_size_actual

            print(
                f"Batch [{batch_idx + 1}/{len(eval_loader)}] "
                f"loss={loss.item():.6f}, accuracy={(predictions == label).float().mean().item():.4f}"
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_accuracy = total_correct / max(total_samples, 1)

    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        inference_peak_allocated = max(peak_allocated - baseline_allocated, 0)
        inference_peak_reserved = max(peak_reserved - baseline_reserved, 0)

        print(f"Peak GPU memory allocated during inference: {bytes_to_mib(inference_peak_allocated):.2f} MiB")
        print(f"Peak GPU memory reserved during inference: {bytes_to_mib(inference_peak_reserved):.2f} MiB")
        print(f"Current GPU memory allocated: {bytes_to_mib(torch.cuda.memory_allocated(device)):.2f} MiB")
        print(f"Current GPU memory reserved: {bytes_to_mib(torch.cuda.memory_reserved(device)):.2f} MiB")
    else:
        print("CUDA is not available, GPU memory statistics are skipped.")

    print("Inference finished successfully.")


if __name__ == "__main__":
    inference()