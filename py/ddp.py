import io
import os
import time
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

if TYPE_CHECKING:
    from multiprocessing import Queue

from batch_producer import DataQueueMessage, batch_producer
from model import Transformer
from params import log_base_path, num_epochs, pad, target_num_tokens_per_batch
from per_process_logs import redirect_stdio
from s3_upload import (
    create_s3_prefix_from_run_id,
    get_checkpoint_files,
    get_s3_config_from_env,
    launch_s3_upload_background,
    validate_s3_config,
)
from training import save_checkpoint


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


def launch_s3_upload_for_epoch(
    epoch: int,
    checkpoint_dir: str,
    run_id: str,
    log_dir: str,
    s3_upload_processes: list[mp.Process],
) -> None:
    """Launch S3 upload for checkpoint files of a specific epoch."""
    try:
        # Get S3 configuration from environment variables
        s3_config = get_s3_config_from_env()

        # Skip if no bucket name is configured (backup check)
        if not s3_config["bucket_name"]:
            print(f"No S3 bucket configured, skipping upload for epoch {epoch}")
            return

        # Get checkpoint files for this epoch
        file_paths = get_checkpoint_files(checkpoint_dir, epoch)

        if not file_paths:
            print(f"No checkpoint files found for epoch {epoch}")
            return

        # Create S3 prefix for this run
        s3_prefix = create_s3_prefix_from_run_id(run_id)

        print(f"Launching S3 upload for epoch {epoch}: {len(file_paths)} files")
        print(f"S3 destination: s3://{s3_config['bucket_name']}/{s3_prefix}/")

        # Launch background upload process
        upload_process = launch_s3_upload_background(
            file_paths=file_paths,
            bucket_name=s3_config["bucket_name"],
            s3_prefix=s3_prefix,
            log_dir=log_dir,
            epoch=epoch,
            aws_access_key_id=s3_config["aws_access_key_id"],
            aws_secret_access_key=s3_config["aws_secret_access_key"],
            aws_region=s3_config["aws_region"] or "eu-north-1",
        )

        print(f"S3 upload process started for epoch {epoch} (PID: {upload_process.pid})")

        # Add to tracking list
        s3_upload_processes.append(upload_process)

    except Exception as e:
        print(f"Failed to launch S3 upload for epoch {epoch}: {e}")
        print("Training will continue without S3 upload for this epoch.")


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


def fd_to_textio(fd: int) -> io.TextIOWrapper:
    return io.TextIOWrapper(
        os.fdopen(fd, "wb", buffering=0), encoding="utf-8", line_buffering=True, errors="replace"
    )


def batch_producer_with_logging(
    target_num_tokens_per_batch: int,
    dataset_base_path: str,
    num_procs: int,
    proc_id: int,
    device_id: str,
    data_queue: "Queue[DataQueueMessage]",
    term_queue: "Queue[None]",
    rng_seed: int,
    log_dir: str,
    epoch: int,
) -> None:
    """Wrapper around batch_producer that handles logging and exceptions."""
    # Setup logging for this process
    console_out, console_err = redirect_stdio(
        os.path.join(log_dir, f"batch_producer_proc_{proc_id}_epoch_{epoch}.log"),
        also_console=False,
    )

    try:
        batch_producer(
            target_num_tokens_per_batch=target_num_tokens_per_batch,
            dataset_base_path=dataset_base_path,
            num_procs=num_procs,
            proc_id=proc_id,
            device_id=device_id,
            data_queue=data_queue,
            term_queue=term_queue,
            rng_seed=rng_seed,
        )
    except Exception as e:
        print(f"Batch producer process {proc_id} failed for epoch {epoch}: {e}")
        raise
    finally:
        # Cleanup any CUDA resources if on GPU
        if "cuda" in device_id:
            torch.cuda.empty_cache()


def train_one_epoch(
    rank: int,
    world_size: int,
    log_dir: str,
    epoch: int,
    model: DDP,
    optimizer: torch.optim.AdamW,
    tqdm_output: io.TextIOWrapper,
    checkpoint_dir: str,
    run_id: str,
) -> float:

    data_queue: mp.Queue[DataQueueMessage] = mp.Queue(maxsize=10)
    term_queue: mp.Queue[None] = mp.Queue()
    rng_seed = 42 + epoch

    # Start batch_producer as separate process
    batch_producer_proc = mp.Process(
        target=batch_producer_with_logging,
        args=(
            target_num_tokens_per_batch,
            "../4_tokens/train",
            world_size,
            rank,
            "cuda:" + str(rank),
            data_queue,
            term_queue,
            rng_seed,
            log_dir,
            epoch,
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
            file=tqdm_output,
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

        # Save checkpoint only on rank 0
        save_checkpoint(model.module, optimizer, epoch, avg_loss, checkpoint_dir)
    else:
        avg_loss = epoch_loss / num_batches

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

    return avg_loss


def train_ddp_worker(
    rank: int,
    world_size: int,
    run_id: str,
    s3_upload_processes: list[mp.Process],
    enable_s3: bool = True,
) -> None:
    """Main training function for each DDP worker process."""
    # Create log directory using run_id and log_base_path
    log_dir = os.path.join(log_base_path, run_id)

    console_out, console_err = redirect_stdio(
        os.path.join(log_dir, f"trainer_proc_{rank}.log"), also_console=rank == 0
    )
    print(f"Running DDP training on rank {rank} of {world_size}")
    print(f"Run ID: {run_id}")
    print(f"Log directory: {log_dir}")

    # Setup DDP
    setup_ddp(rank, world_size)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)  # type: ignore
    torch.autograd.profiler.emit_nvtx(False)  # type: ignore
    torch.set_float32_matmul_precision("high")

    # Setup checkpoint directory
    checkpoint_dir = "../5_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Create model
        model = create_ddp_model(rank)

        model.compile(dynamic=True)  # type: ignore

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training loop
        model.train()  # type: ignore
        for epoch in range(num_epochs):
            train_one_epoch(
                rank,
                world_size,
                log_dir,
                epoch,
                model,
                optimizer,
                console_err,
                checkpoint_dir,
                run_id,
            )

            # Launch S3 upload on rank 0 after each epoch (if enabled)
            if rank == 0 and enable_s3:
                launch_s3_upload_for_epoch(
                    epoch, checkpoint_dir, run_id, log_dir, s3_upload_processes
                )

    finally:
        cleanup_ddp()


def launch_ddp_training(world_size: int | None = None, enable_s3: bool = True) -> None:
    """Launch DDP training across multiple GPUs."""

    # Generate run ID with timestamp
    run_id = f"ddp_{time.strftime('%Y%m%d_%H%M%S')}"

    print("Initializing DDP training")
    print(f"Run ID: {run_id}")

    # Validate S3 configuration early to fail fast if misconfigured
    if enable_s3:
        try:
            s3_config = validate_s3_config()
            print(f"✓ S3 uploads enabled: s3://{s3_config['bucket_name']}/runs/{run_id}/")
        except (ValueError, RuntimeError) as e:
            print("✗ S3 configuration validation failed:")
            print(f"  {e}")
            print("Please fix S3 configuration or set enable_s3=False to continue.")
            raise SystemExit(1)
    else:
        print("⚠ S3 uploads disabled")

    # Set multiprocessing start method for CUDA tensor sharing
    mp.set_start_method("spawn", force=True)

    if world_size is None:
        world_size = torch.cuda.device_count()

    assert world_size is not None  # Type hint for mypy

    if world_size < 2:
        print("Warning: DDP requires at least 2 GPUs. Falling back to single GPU training.")
        return

    print(f"Launching DDP training with {world_size} processes")

    # Create list to track S3 upload processes
    s3_upload_processes: list[mp.Process] = []

    # Spawn training processes
    mp.spawn(  # type: ignore
        train_ddp_worker,
        args=(world_size, run_id, s3_upload_processes, enable_s3),
        nprocs=world_size,
        join=True,
    )

    # Wait for all S3 upload processes to complete
    if enable_s3 and s3_upload_processes:
        print(f"Waiting for {len(s3_upload_processes)} S3 upload processes to complete...")
        for process in s3_upload_processes:
            if process.is_alive():
                process.join()
        print("All S3 uploads completed.")


# Example usage
if __name__ == "__main__":
    # Launch DDP training with dummy data
    launch_ddp_training(world_size=2)
