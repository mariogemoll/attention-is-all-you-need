import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from batch_producer import DataQueueMessage, batch_producer
from model import Transformer
from params import pad, target_num_tokens_per_batch


class DummyProgressBar:
    def __init__(self) -> None:
        pass

    def set_postfix(self, info: dict[str, str]) -> None:
        pass

    def update(self, n: int = 1) -> None:
        pass


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
    rank: int,
    world_size: int,
    epoch: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
) -> None:

    data_queue: mp.Queue[DataQueueMessage] = mp.Queue(maxsize=10)
    term_queue: mp.Queue[None] = mp.Queue()
    rng_seed = 42 + epoch

    # Start batch_producer as separate process
    batch_producer_proc = mp.Process(
        target=batch_producer,
        args=(
            target_num_tokens_per_batch,
            "../4_tokens/train",
            world_size,
            rank,
            "cuda:" + str(rank),
            data_queue,
            term_queue,
            rng_seed,
        ),
    )
    batch_producer_proc.start()

    initial_msg = data_queue.get()
    assert initial_msg["type"] == "start"
    num_batches = initial_msg["num_batches"]
    del initial_msg

    epoch_loss = 0.0

    pbar: tqdm[int] | DummyProgressBar
    if rank == 0:
        pbar = tqdm(
            range(num_batches),
            desc=f"Epoch {epoch}",
            bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        )
    else:
        pbar = DummyProgressBar()

    start_time = time.time()
    for batch_idx in range(num_batches):

        msg = data_queue.get()
        assert msg["type"] == "batch"
        enc_input, dec_input, dec_target = msg["data"]

        memory = model.module.encode(enc_input)
        output = model.module.decode(enc_input, memory, dec_input)

        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)
        loss = loss_fn(output.reshape(-1, output.size(-1)), dec_target.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        # Explicitly delete tensors and synchronize CUDA
        del enc_input, dec_input, dec_target, output, memory, loss, msg

        if rank == 0:
            estimated_total_time, time_info_str = time_info(
                num_batches, batch_idx + 1, start_time, time.time()
            )
            pbar.update(1)
            pbar.set_postfix({"time": time_info_str, "loss": f"{current_loss:.2f} "})

    if rank == 0:
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

    # Synchronize all CUDA operations before cleanup
    torch.cuda.synchronize(rank)

    # Do garbage collection
    torch.cuda.empty_cache()

    # Signal termination and wait for producer process to finish
    term_queue.put(None)
    batch_producer_proc.join(timeout=10)  # Wait up to 10 seconds

    # Force terminate if still alive
    if batch_producer_proc.is_alive():
        print(f"Rank {rank}: Forcefully terminating batch producer process")
        batch_producer_proc.terminate()
        batch_producer_proc.join()


def train_ddp_worker(rank: int, world_size: int, epochs: int = 10) -> None:
    """Main training function for each DDP worker process."""
    print(f"Running DDP training on rank {rank} of {world_size}")

    # Setup DDP
    setup_ddp(rank, world_size)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)  # type: ignore
    torch.autograd.profiler.emit_nvtx(False)  # type: ignore
    torch.set_float32_matmul_precision("high")

    try:
        # Create model
        model = create_ddp_model(rank)

        model.compile(dynamic=True)  # type: ignore

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training loop with dummy data
        model.train()  # type: ignore
        for epoch in range(epochs):
            train_one_epoch(rank, world_size, epoch, model, optimizer)

    finally:
        cleanup_ddp()


def launch_ddp_training(
    world_size: int | None = None, epochs: int = 10, num_batches: int = 100
) -> None:
    """Launch DDP training across multiple GPUs."""
    # Set multiprocessing start method for CUDA tensor sharing
    mp.set_start_method("spawn", force=True)

    if world_size is None:
        world_size = torch.cuda.device_count()

    assert world_size is not None  # Type hint for mypy

    if world_size < 2:
        print("Warning: DDP requires at least 2 GPUs. Falling back to single GPU training.")
        return

    print(f"Launching DDP training with {world_size} processes")

    # Spawn training processes
    mp.spawn(  # type: ignore
        train_ddp_worker, args=(world_size, epochs), nprocs=world_size, join=True
    )


# Example usage
if __name__ == "__main__":
    # Launch DDP training with dummy data
    launch_ddp_training(world_size=2, epochs=3, num_batches=50)
