import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Transformer


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set the GPU device for this process
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def create_ddp_model(rank: int) -> DDP:
    """Create and wrap model with DDP."""
    model = Transformer()
    model = model.to(rank)  # Move model to the specific GPU

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model


def create_dummy_batch(
    batch_size: int = 32, seq_len: int = 128, vocab_size: int = 32000, device: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create dummy input tensors for testing DDP setup."""
    src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    return src, tgt


def train_ddp_worker(rank: int, world_size: int, epochs: int = 10, num_batches: int = 100) -> None:
    """Main training function for each DDP worker process."""
    print(f"Running DDP training on rank {rank} of {world_size}")

    # Setup DDP
    setup_ddp(rank, world_size)

    try:
        # Create model
        model = create_ddp_model(rank)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training loop with dummy data
        model.train()  # type: ignore
        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                # Create dummy batch
                src, tgt = create_dummy_batch(batch_size=32, device=rank)

                # Forward pass
                memory = model.module.encode(src)
                output = model.module.decode(src, memory, tgt[:, :-1])

                # Calculate loss
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is pad token
                loss = loss_fn(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if rank == 0 and batch_idx % 20 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            if rank == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

    finally:
        cleanup_ddp()


def launch_ddp_training(
    world_size: int | None = None, epochs: int = 10, num_batches: int = 100
) -> None:
    """Launch DDP training across multiple GPUs."""
    if world_size is None:
        world_size = torch.cuda.device_count()

    assert world_size is not None  # Type hint for mypy

    if world_size < 2:
        print("Warning: DDP requires at least 2 GPUs. Falling back to single GPU training.")
        return

    print(f"Launching DDP training with {world_size} processes")

    # Spawn training processes
    mp.spawn(  # type: ignore
        train_ddp_worker, args=(world_size, epochs, num_batches), nprocs=world_size, join=True
    )


# Example usage
if __name__ == "__main__":
    # Launch DDP training with dummy data
    launch_ddp_training(world_size=2, epochs=3, num_batches=50)
