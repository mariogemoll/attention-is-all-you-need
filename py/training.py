from __future__ import annotations

import os
import re
import time
from typing import Any, Dict

import torch
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import params as training_params
from batch_producer import DataQueueMessage
from batching import EpochBatches
from buckets import BucketedDataset
from model import Transformer
from params import pad, target_num_tokens_per_batch
from tensors import get_tensors

CHECKPOINT_PARAM_TYPES = (int, float, str, bool)


def _collect_params_snapshot() -> dict[str, int | float | str | bool]:
    snapshot: dict[str, int | float | str | bool] = {}
    for name in dir(training_params):
        if name.startswith("_"):
            continue
        value = getattr(training_params, name)
        if isinstance(value, CHECKPOINT_PARAM_TYPES):
            snapshot[name] = value
    return snapshot


def save_checkpoint(
    model: Transformer,
    optimizer: AdamW,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
) -> None:
    """Save training checkpoint with model weights and metadata in separate files."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model weights separately
    epoch_str = f"{epoch:04d}"
    model_weights_file = f"model_{epoch_str}.pt"
    model_weights_path = os.path.join(checkpoint_dir, model_weights_file)
    torch.save(model.state_dict(), model_weights_path)

    # Save checkpoint metadata (without model weights)
    if scheduler is not None:
        scheduler_state_dict = scheduler.state_dict()
    else:
        scheduler_state_dict = None

    checkpoint_metadata: Dict[str, Any] = {
        "epoch": epoch,
        "model_weights_file": model_weights_file,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "scheduler_state_dict": scheduler_state_dict,
        "params_snapshot": _collect_params_snapshot(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch_str}.pt")
    torch.save(checkpoint_metadata, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Model weights saved: {model_weights_path}")


def load_checkpoint(
    model: Transformer,
    optimizer: AdamW,
    checkpoint_path: str,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    strict_params: bool = True,
) -> tuple[int, float]:
    """Load training checkpoint and restore model, optimizer state, and RNG state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model weights from separate file
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model_weights_file = checkpoint["model_weights_file"]
    model_weights_path = os.path.join(checkpoint_dir, model_weights_file)

    model_state_dict = torch.load(model_weights_path, map_location="cpu")
    model.load_state_dict(model_state_dict)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state if provided
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif scheduler is not None and scheduler_state is None:
        print(
            "Warning: Scheduler provided but checkpoint lacks scheduler state."
            " Scheduler will continue from current state."
        )

    # Restore RNG states
    torch.set_rng_state(checkpoint["rng_state"])
    if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    stored_params = checkpoint.get("params_snapshot")
    if stored_params:
        current_params = _collect_params_snapshot()
        mismatches = {}
        for key, stored_value in stored_params.items():
            current_value = current_params.get(key)
            if current_value != stored_value:
                mismatches[key] = (stored_value, current_value)
        if mismatches:
            mismatch_details = ", ".join(
                f"{key}: checkpoint={stored} current={current}"
                for key, (stored, current) in mismatches.items()
            )
            message = (
                "Checkpoint parameter mismatch detected. "
                f"The following settings differ from current params: {mismatch_details}"
            )
            if strict_params:
                raise RuntimeError(message)
            else:
                print(f"Warning: {message}")

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded: {checkpoint_path} (epoch {epoch}, loss: {loss:.4f})")
    print(f"Model weights loaded: {model_weights_path}")
    return epoch, loss


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint file by epoch number."""
    if not os.path.isdir(checkpoint_dir):
        return None

    latest_epoch = -1
    latest_path: str | None = None

    pattern = re.compile(r"^checkpoint(?:_epoch)?_(\d+)\.pt$")

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if not match:
            continue

        try:
            epoch_num = int(match.group(1))
        except ValueError:
            continue

        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = os.path.join(checkpoint_dir, filename)

    return latest_path


def clean_up_old_checkpoints(checkpoint_dir: str, keep_last: int = 3) -> None:
    """Remove old checkpoint files, keeping only the last N checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of most recent checkpoints to keep (default: 3)
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Find all numbered checkpoint files
    checkpoint_files = []
    model_files = []

    for filename in os.listdir(checkpoint_dir):
        checkpoint_match = re.match(r"^checkpoint(?:_epoch)?_(\d+)\.pt$", filename)
        if checkpoint_match:
            try:
                epoch_num = int(checkpoint_match.group(1))
            except ValueError:
                continue
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            checkpoint_files.append((epoch_num, checkpoint_path))
            continue

        model_match = re.match(r"^model(?:_epoch)?_(\d+)\.pt$", filename)
        if model_match:
            try:
                epoch_num = int(model_match.group(1))
            except ValueError:
                continue
            model_path = os.path.join(checkpoint_dir, filename)
            model_files.append((epoch_num, model_path))

    # Sort by epoch number and keep only the most recent ones
    checkpoint_files.sort(key=lambda x: x[0])
    model_files.sort(key=lambda x: x[0])

    # Remove old checkpoint files (keep the latest N)
    if len(checkpoint_files) > keep_last:
        files_to_remove = checkpoint_files[:-keep_last]
        for epoch_num, filepath in files_to_remove:
            try:
                os.remove(filepath)
                print(f"Removed old checkpoint: {os.path.basename(filepath)}")
            except OSError as e:
                print(f"Warning: Could not remove {filepath}: {e}")

    # Remove old model weight files (keep the latest N)
    if len(model_files) > keep_last:
        files_to_remove = model_files[:-keep_last]
        for epoch_num, filepath in files_to_remove:
            try:
                os.remove(filepath)
                print(f"Removed old model weights: {os.path.basename(filepath)}")
            except OSError as e:
                print(f"Warning: Could not remove {filepath}: {e}")


def init(
    device: torch.device, checkpoint_dir: str, resume_from_checkpoint: bool = True
) -> tuple[Transformer, AdamW, CrossEntropyLoss, SummaryWriter, int]:
    model = Transformer().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-10)
    criterion = CrossEntropyLoss(ignore_index=pad)
    writer = SummaryWriter()  # type: ignore

    start_epoch = 1

    if resume_from_checkpoint:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path:
            try:
                last_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
                start_epoch = last_epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch")
        else:
            print("No checkpoint found, starting training from scratch")

    return model, optimizer, criterion, writer, start_epoch


def time_info(
    batches_total: int, batches_processed: int, start_time: float, current_time: float
) -> tuple[float, str]:
    elapsed_time = current_time - start_time
    elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    estimated_total_time = (elapsed_time / batches_processed) * batches_total
    estimated_total_time_str = time.strftime("%M:%S", time.gmtime(estimated_total_time))
    estimated_remaining_time = estimated_total_time - elapsed_time
    estimated_remaining_time_str = time.strftime("%M:%S", time.gmtime(estimated_remaining_time))
    return (
        estimated_total_time,
        f"{elapsed_time_str}|{estimated_remaining_time_str}|{estimated_total_time_str}",
    )


def train_one_epoch(
    model: Transformer,
    optimizer: AdamW,
    criterion: CrossEntropyLoss,
    writer: SummaryWriter,
    data_queue: mp.Queue[DataQueueMessage],
    epoch: int,
) -> float:

    model.train()

    initial_msg = data_queue.get()
    assert initial_msg["type"] == "start"
    num_batches_total = initial_msg["num_batches"]

    pbar = tqdm(
        range(num_batches_total),
        desc=f"Epoch {epoch}",
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
    )
    total_loss = 0.0

    # Get the current timestamp
    start_time = time.time()
    for batch_idx in pbar:

        global_step = (epoch - 1) * num_batches_total + batch_idx + 1

        batch_start_time = time.time()
        msg = data_queue.get()
        assert msg["type"] == "batch"
        enc_input, dec_input, dec_target = msg["data"]
        tensor_acquisition = time.time() - batch_start_time

        optimizer.zero_grad()
        memory = model.encode(enc_input)
        out = model.decode(enc_input, memory, dec_input)
        loss = criterion(out.view(-1, out.size(-1)), dec_target.view(-1))
        loss.backward()

        # Log training loss
        current_loss = loss.item()

        del enc_input, dec_input, dec_target, memory, out, loss

        total_loss += current_loss

        estimated_total_time, time_info_str = time_info(
            num_batches_total, batch_idx + 1, start_time, time.time()
        )

        writer.add_scalar("time/tensor_aquisition", tensor_acquisition, global_step)  # type: ignore
        writer.add_scalar("time/estimated_total", estimated_total_time, global_step)  # type: ignore
        writer.add_scalar("loss/batch/train", current_loss, global_step)  # type: ignore
        pbar.set_postfix({"time": time_info_str, "loss": f"{current_loss:.2f} "})
        optimizer.step()

    # Log average training loss for the epoch
    avg_train_loss = total_loss / num_batches_total
    writer.add_scalar("loss/epoch/train", avg_train_loss, epoch)  # type: ignore
    print(f"Average training loss for epoch {epoch}: {avg_train_loss:.4f}")
    return avg_train_loss


def evaluate(
    device: torch.device,
    valset: BucketedDataset,
    model: Transformer,
    criterion: CrossEntropyLoss,
    writer: SummaryWriter,
    epoch: int,
) -> float:
    model.eval()
    val_batches = EpochBatches(
        1,
        0,
        valset.bucket_index_file,
        target_num_tokens_per_batch,
        rng_seed=42,
        full_batches_only=True,
    )
    with torch.no_grad():
        losses = torch.zeros(len(val_batches))
        pbar = tqdm(val_batches, desc=f"Evaluating Epoch {epoch}")
        for i, (batch_id, entry_ids) in enumerate(pbar):
            seq_len = (batch_id + 1) * valset.step_size
            enc_input, dec_input, dec_target = get_tensors(seq_len, valset.dataset, entry_ids)
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            memory = model.encode(enc_input)
            out = model.decode(enc_input, memory, dec_input)
            loss = criterion(out.view(-1, out.size(-1)), dec_target.view(-1))
            losses[i] = loss
            pbar.set_postfix({"loss": loss.item()})
    epoch_loss = losses.mean().item()
    writer.add_scalar("loss/epoch/val", epoch_loss, epoch)  # type: ignore
    print(f"Validation loss for epoch {epoch}: {epoch_loss}")
    return epoch_loss
