import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from batching import EpochBatches
from data import BucketedDataset, get_tensors
from model import Transformer
from params import pad
from util import get_device


def save_checkpoint(
    model: Transformer,
    optimizer: AdamW,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
) -> None:
    """Save training checkpoint with model weights and metadata in separate files."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model weights separately
    model_weights_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), model_weights_path)

    # Save checkpoint metadata (without model weights)
    checkpoint_metadata = {
        "epoch": epoch,
        "model_weights_file": f"model_epoch_{epoch}.pt",
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint_metadata, checkpoint_path)

    # Also save as latest checkpoint and model weights
    latest_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    latest_model_path = os.path.join(checkpoint_dir, "model_latest.pt")

    checkpoint_metadata["model_weights_file"] = "model_latest.pt"
    torch.save(checkpoint_metadata, latest_checkpoint_path)
    torch.save(model.state_dict(), latest_model_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Model weights saved: {model_weights_path}")


def load_checkpoint(
    model: Transformer, optimizer: AdamW, checkpoint_path: str
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

    # Restore RNG states
    torch.set_rng_state(checkpoint["rng_state"])
    if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded: {checkpoint_path} (epoch {epoch}, loss: {loss:.4f})")
    print(f"Model weights loaded: {model_weights_path}")
    return epoch, loss


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint file."""
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    return None


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
        if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
            try:
                epoch_num = int(filename.replace("checkpoint_epoch_", "").replace(".pt", ""))
                checkpoint_path = os.path.join(checkpoint_dir, filename)
                checkpoint_files.append((epoch_num, checkpoint_path))
            except ValueError:
                continue

        elif filename.startswith("model_epoch_") and filename.endswith(".pt"):
            try:
                epoch_num = int(filename.replace("model_epoch_", "").replace(".pt", ""))
                model_path = os.path.join(checkpoint_dir, filename)
                model_files.append((epoch_num, model_path))
            except ValueError:
                continue

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
    checkpoint_dir: str,
    resume_from_checkpoint: bool = True,
) -> tuple[torch.device, Transformer, AdamW, CrossEntropyLoss, SummaryWriter, int]:
    device = get_device()
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

    return device, model, optimizer, criterion, writer, start_epoch


def train_one_epoch(
    device: torch.device,
    trainset: BucketedDataset,
    model: Transformer,
    optimizer: AdamW,
    criterion: CrossEntropyLoss,
    writer: SummaryWriter,
    epoch: int,
) -> float:
    model.train()
    train_batches = EpochBatches(
        trainset.bucket_index_file,
        target_tokens_per_batch=25000,
        shuffle_within_buckets=True,
        shuffle_batches=True,
        random_seed=42,
        full_batches_only=True,
    )

    pbar = tqdm(train_batches, desc=f"Epoch {epoch}")
    total_loss = 0.0
    num_batches = 0

    for batch_id, entry_ids in pbar:
        seq_len = (batch_id + 1) * trainset.step_size
        enc_input, dec_input, dec_target = get_tensors(
            trainset.index_file, trainset.data_file, trainset.data_file_size, seq_len, entry_ids
        )
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        dec_target = dec_target.to(device)
        optimizer.zero_grad()
        memory = model.encode(enc_input)
        out = model.decode(enc_input, memory, dec_input)
        loss = criterion(out.view(-1, out.size(-1)), dec_target.view(-1))
        loss.backward()

        # Log training loss
        current_loss = loss.item()
        total_loss += current_loss
        num_batches += 1
        global_step = (epoch - 1) * len(train_batches) + num_batches
        writer.add_scalar("loss/batch/train", current_loss, global_step)  # type: ignore
        pbar.set_postfix({"loss": current_loss})
        optimizer.step()

    # Log average training loss for the epoch
    avg_train_loss = total_loss / num_batches
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
        valset.bucket_index_file,
        target_tokens_per_batch=25000,
        shuffle_within_buckets=False,
        shuffle_batches=False,
        random_seed=42,
        full_batches_only=True,
    )
    with torch.no_grad():
        losses = torch.zeros(len(val_batches))
        pbar = tqdm(val_batches, desc=f"Evaluating Epoch {epoch}")
        for i, (batch_id, entry_ids) in enumerate(pbar):
            seq_len = (batch_id + 1) * valset.step_size
            enc_input, dec_input, dec_target = get_tensors(
                valset.index_file, valset.data_file, valset.data_file_size, seq_len, entry_ids
            )
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
