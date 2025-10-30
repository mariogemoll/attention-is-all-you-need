import argparse
import math
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import open_dataset
from model import Transformer
from params import (
    aiayn_tokens_per_step,
    log_base_path,
    max_seq_len,
    pad,
    target_num_tokens_per_batch,
)
from tensors import get_tensors
from training import save_checkpoint
from util import get_device


def load_single_batch(
    dataset_path: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load all entries from a single-batch dataset.

    Args:
        dataset_path: Path to the single-batch dataset (without extension)
        device: Device to load tensors to

    Returns:
        Tuple of (enc_input, dec_input, dec_target) tensors
    """
    with open_dataset(dataset_path) as dataset:
        num_entries = dataset.num_entries
        if num_entries == 0:
            raise RuntimeError(f"No entries in dataset: {dataset_path}")

        # Load all entries from the dataset
        entry_ids = list(range(num_entries))

        # Use max_seq_len from params as the sequence length
        seq_len = max_seq_len
        enc_input, dec_input, dec_target = get_tensors(
            seq_len,
            dataset,
            entry_ids,
        )

        print(f"Loaded single batch from {dataset_path}")
        print(f"  Number of entries: {num_entries}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Batch shape: {enc_input.shape}")

    return (
        enc_input.to(device, non_blocking=True),
        dec_input.to(device, non_blocking=True),
        dec_target.to(device, non_blocking=True),
    )


def train_single_batch(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    writer: SummaryWriter | None,
    steps: int,
    device: torch.device,
    dataset_path: str,
) -> float:
    enc_input, dec_input, dec_target = load_single_batch(dataset_path, device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad).to(device)

    progress = tqdm(
        range(steps),
        desc="Single-batch training",
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
    )

    running_loss = 0.0
    start_time = time.time()

    for step_idx in range(steps):
        step_start = time.time()

        memory = model.encode(enc_input)
        output = model.decode(enc_input, memory, dec_input)

        loss = loss_fn(output.reshape(-1, output.size(-1)), dec_target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = loss.item()
        running_loss += loss_value

        del memory, output, loss

        step_duration = time.time() - step_start
        elapsed_time = time.time() - start_time
        avg_step_time = elapsed_time / (step_idx + 1)
        remaining_steps = steps - (step_idx + 1)
        eta_seconds = remaining_steps * avg_step_time
        eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"

        if writer is not None:
            global_step = step_idx + 1
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar(  # type: ignore[no-untyped-call]
                "loss/batch/train", loss_value, global_step
            )
            writer.add_scalar(  # type: ignore[no-untyped-call]
                "time/step_duration", step_duration, global_step
            )
            writer.add_scalar("lr/batch", current_lr, global_step)  # type: ignore[no-untyped-call]

        progress.update(1)
        progress.set_postfix({"ETA": eta_str, "loss": f"{loss_value:.2f}"})

    avg_loss = running_loss / max(steps, 1)

    print(f"Completed {steps} steps on a single batch. Average loss: {avg_loss:.4f}")
    if writer is not None:
        writer.add_scalar("loss/avg_single_batch", avg_loss, steps)  # type: ignore[no-untyped-call]
        writer.flush()  # type: ignore[no-untyped-call]

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return avg_loss


def run_single_batch_training(steps: int, dataset_path: str) -> None:
    run_id = f"single_batch_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(log_base_path, run_id)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Starting single-batch training run {run_id}")
    print(f"Log directory: {log_dir}")
    print(f"Dataset: {dataset_path}")

    writer = SummaryWriter(  # type: ignore[no-untyped-call]
        log_dir=os.path.join(log_dir, "tensorboard")
    )

    device = get_device()
    print(f"Using device: {device}")

    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision("high")

    model: Transformer = Transformer().to(device)
    model.compile(dynamic=True)  # type: ignore[no-untyped-call]
    model.train()

    tokens_per_step = target_num_tokens_per_batch
    if tokens_per_step <= 0:
        raise RuntimeError("tokens_per_step must be positive")

    base_lr = (target_num_tokens_per_batch / float(aiayn_tokens_per_step)) * 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    warmup_steps = max(1, min(steps, math.ceil(steps * 0.1))) if steps > 0 else 1
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if steps > 0:

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float((current_step + 1) / max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"Will perform {steps} optimizer steps on a single batch")
    print(f"Base learning rate: {base_lr:.6f}")
    if scheduler is not None:
        print(f"Warmup steps: {warmup_steps}")

    avg_loss = train_single_batch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        steps=steps,
        device=device,
        dataset_path=dataset_path,
    )

    checkpoint_dir = os.path.join("../5_checkpoints", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_checkpoint(
        model,
        optimizer,
        epoch=1,
        loss=avg_loss,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
    )
    print(f"Checkpoint saved to {checkpoint_dir}")

    writer.close()  # type: ignore[no-untyped-call]


def launch_single_batch_training(
    steps: int = 200, dataset_path: str = "../4_tokens/single_batch"
) -> None:
    if steps <= 0:
        raise ValueError("steps must be positive")

    print("Initializing single-batch training")
    run_single_batch_training(steps, dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-batch overfitting test")
    parser.add_argument("--steps", type=int, default=2000, help="Optimizer steps to run")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../4_tokens/single_batch",
        help="Path to single-batch dataset (without extension). Default: ../4_tokens/single_batch",
    )
    args = parser.parse_args()

    launch_single_batch_training(steps=args.steps, dataset_path=args.dataset)


if __name__ == "__main__":
    main()
