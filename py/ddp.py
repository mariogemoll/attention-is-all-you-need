import io
import math
import os
import socket
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if TYPE_CHECKING:
    from multiprocessing import Queue

from batch_producer import DataQueueMessage, batch_producer
from batching import EpochBatches
from buckets import open_buckets
from lr_schedules import cosine_lr
from model import Transformer
from model_utils import print_model_parameters
from params import (
    aiayn_tokens_per_step,
    aiayn_warmup_steps,
    checkpoints_to_keep,
    log_base_path,
    pad,
    target_num_processed_tokens,
    target_num_tokens_per_batch,
    train_dataset_path,
)
from per_process_logs import redirect_stdio
from s3_upload import (
    create_s3_prefix_from_run_id,
    get_checkpoint_files,
    get_s3_config_from_env,
    launch_s3_upload_background,
    validate_s3_config,
)
from training import (
    clean_up_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class DummyProgressBar:
    def __init__(self) -> None:
        pass

    def set_postfix(self, info: dict[str, str]) -> None:
        pass

    def update(self, n: int = 1) -> None:
        pass


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port: int = s.getsockname()[1]
    return port


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    # Port should be set before spawning processes
    if "MASTER_PORT" not in os.environ:
        raise RuntimeError("MASTER_PORT must be set before calling setup_ddp")

    print(f"Rank {rank}: Initializing process group on port {os.environ['MASTER_PORT']}")

    # Initialize the process group with timeout
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )

    print(f"Rank {rank}: Process group initialized successfully")

    # Set the GPU device for this process
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: Set CUDA device to {rank}")


def launch_s3_upload_for_epoch(
    epoch: int,
    checkpoint_dir: str,
    run_id: str,
    log_dir: str,
    s3_upload_processes: list[tuple[mp.Process, str]],
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
        s3_upload_processes.append((upload_process, f"epoch {epoch}"))

    except Exception as e:
        print(f"Failed to launch S3 upload for epoch {epoch}: {e}")
        print("Training will continue without S3 upload for this epoch.")


def cleanup_ddp() -> None:
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def create_ddp_model(rank: int) -> DDP:
    """Create and wrap model with DDP."""
    model = Transformer()

    # Print parameters on rank 0 only
    if rank == 0:
        print_model_parameters(model)

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
    writer: SummaryWriter | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
) -> float:

    data_queue: mp.Queue[DataQueueMessage] = mp.Queue(maxsize=10)
    term_queue: mp.Queue[None] = mp.Queue()
    rng_seed = 42 + epoch

    # Start batch_producer as separate process
    batch_producer_proc = mp.Process(
        target=batch_producer_with_logging,
        args=(
            target_num_tokens_per_batch,
            train_dataset_path,
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
        batch_start_time = time.time()
        msg = data_queue.get()
        assert msg["type"] == "batch"
        enc_input, dec_input, dec_target = msg["data"]
        tensor_acquisition = time.time() - batch_start_time

        memory = model.module.encode(enc_input)
        output = model.module.decode(enc_input, memory, dec_input)

        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)
        loss = loss_fn(output.reshape(-1, output.size(-1)), dec_target.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        current_loss = loss.item()
        epoch_loss += current_loss

        # Explicitly delete tensors and synchronize CUDA
        del enc_input, dec_input, dec_target, output, memory, loss, msg

        if rank == 0:
            estimated_total_time, time_info_str = time_info(
                num_batches, batch_idx + 1, start_time, time.time()
            )
            if writer is not None:
                global_step = (epoch - 1) * num_batches + batch_idx + 1
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar(  # type: ignore[no-untyped-call]
                    "time/tensor_aquisition", tensor_acquisition, global_step
                )
                writer.add_scalar(  # type: ignore[no-untyped-call]
                    "time/estimated_total", estimated_total_time, global_step
                )
                writer.add_scalar(  # type: ignore[no-untyped-call]
                    "loss/batch/train", current_loss, global_step
                )
                writer.add_scalar(  # type: ignore[no-untyped-call]
                    "lr/batch", current_lr, global_step
                )
                writer.add_scalar(  # type: ignore[no-untyped-call]
                    "gradient_norm/batch", grad_norm.item(), global_step
                )
            pbar.update(1)
            pbar.set_postfix({"time": time_info_str, "loss": f"{current_loss:.2f} "})

    avg_loss = epoch_loss / num_batches if num_batches else 0.0

    if rank == 0:
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        if writer is not None:
            writer.add_scalar("loss/epoch/train", avg_loss, epoch)  # type: ignore[no-untyped-call]
            writer.flush()  # type: ignore[no-untyped-call]

        # Save checkpoint only on rank 0
        save_checkpoint(
            model.module,
            optimizer,
            epoch,
            avg_loss,
            checkpoint_dir,
            scheduler=scheduler,
        )

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
    enable_s3: bool = True,
    resume_from_checkpoint: bool = True,
) -> None:
    """Main training function for each DDP worker process."""
    # Create log directory using run_id and log_base_path
    log_dir = os.path.join(log_base_path, run_id)
    os.makedirs(log_dir, exist_ok=True)

    console_out, console_err = redirect_stdio(
        os.path.join(log_dir, f"trainer_proc_{rank}.log"), also_console=rank == 0
    )
    print(f"Running DDP training on rank {rank} of {world_size}")
    print(f"Run ID: {run_id}")
    print(f"Log directory: {log_dir}")

    print(f"Rank {rank}: Setting up TensorBoard writer...")
    writer: SummaryWriter | None = None
    if rank == 0:
        writer_log_dir = os.path.join(log_dir, "tensorboard")
        print(f"Rank {rank}: Creating SummaryWriter at {writer_log_dir}")
        writer = SummaryWriter(log_dir=writer_log_dir)  # type: ignore[no-untyped-call]
        print(f"Rank {rank}: SummaryWriter created")
    else:
        print(f"Rank {rank}: Skipping SummaryWriter (not rank 0)")

    print(f"Rank {rank}: About to call setup_ddp()")
    # Setup DDP
    setup_ddp(rank, world_size)
    print(f"Rank {rank}: setup_ddp() completed")

    print(f"Rank {rank}: Setting torch configurations...")
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)  # type: ignore
    torch.autograd.profiler.emit_nvtx(False)  # type: ignore
    torch.set_float32_matmul_precision("high")

    # Setup checkpoint directory
    checkpoint_dir = "../5_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    s3_upload_processes: list[tuple[mp.Process, str]] = []
    cleanup_processes: list[tuple[mp.Process, str]] = []

    try:
        # Create model
        print(f"Rank {rank}: Creating DDP model...")
        model = create_ddp_model(rank)
        print(f"Rank {rank}: DDP model created")

        print(f"Rank {rank}: Compiling model...")
        model.compile(dynamic=True)  # type: ignore
        print(f"Rank {rank}: Model compiled")

        training_config: list[float] = [0.0] * 6
        if rank == 0:
            print(f"Rank {rank}: Opening buckets to calculate training schedule...")
            try:
                with open_buckets(train_dataset_path) as train_buckets:
                    epoch_batches_preview = EpochBatches(
                        num_procs=world_size,
                        proc_id=0,
                        bucket_index_file=train_buckets.bucket_index_file,
                        target_num_tokens_per_batch=target_num_tokens_per_batch,
                        rng_seed=42,
                        full_batches_only=True,
                    )
                    batches_per_epoch = len(epoch_batches_preview)
            except FileNotFoundError as e:
                print(f"Rank {rank}: ERROR - {e}")
                print(f"Rank {rank}: Please add training data")
                # Abort the process group to avoid hanging other ranks
                if dist.is_initialized():
                    print(f"Rank {rank}: Aborting distributed process group due to error...")
                    # Send sentinel values to unblock other ranks
                    training_config = [-1.0] * 6  # Use -1 to signal error
                    try:
                        dist.broadcast_object_list(training_config, src=0)
                    except Exception:
                        pass  # If broadcast fails, that's ok
                raise

            if batches_per_epoch <= 0:
                raise RuntimeError(
                    "No batches available for training; check dataset configuration."
                )

            tokens_per_step = target_num_tokens_per_batch * world_size
            if tokens_per_step <= 0:
                raise RuntimeError("Tokens per step must be positive. Check configuration.")

            target_steps = max(1, math.ceil(target_num_processed_tokens / tokens_per_step))
            epochs_total = max(1, math.ceil(target_steps / batches_per_epoch))

            warmup_steps_target = math.ceil(
                (aiayn_tokens_per_step * aiayn_warmup_steps) / tokens_per_step
            )
            warmup_steps = max(1, min(target_steps, warmup_steps_target))
            warmup_epochs = min(epochs_total, max(1, math.ceil(warmup_steps / batches_per_epoch)))

            total_steps = epochs_total * batches_per_epoch
            training_config = [
                float(batches_per_epoch),
                float(epochs_total),
                float(warmup_steps),
                float(total_steps),
                float(target_steps),
                float(warmup_epochs),
            ]

            print(
                "Training schedule:",
                f"batches/epoch={batches_per_epoch}",
                f"tokens/step={tokens_per_step}",
                f"target_tokens={target_num_processed_tokens}",
                f"target_steps={target_steps}",
                f"epochs={epochs_total}",
                f"total_steps={total_steps}",
                f"warmup_steps={warmup_steps}",
                f"warmup_epochs≈{warmup_epochs}",
            )
        else:
            print(f"Rank {rank}: Waiting for training config from rank 0...")

        dist.broadcast_object_list(training_config, src=0)
        print(f"Rank {rank}: Received training config: {training_config}")

        # Check if rank 0 sent an error signal
        if training_config[0] < 0:
            print(f"Rank {rank}: Received error signal from rank 0, exiting...")
            raise RuntimeError("Rank 0 encountered an error during initialization")

        batches_per_epoch, epochs_total, warmup_steps, total_steps, target_steps, warmup_epochs = [
            int(x) for x in training_config
        ]

        if batches_per_epoch <= 0 or total_steps <= 0:
            raise RuntimeError("Invalid training schedule received from rank 0.")

        print(f"Rank {rank}: Creating optimizer and scheduler...")
        # Match overfit_single_batch.py: base multiplier 1e-4, with linear scaling for world_size
        base_lr = (target_num_tokens_per_batch / float(aiayn_tokens_per_step)) * 1e-4 * world_size
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        schedule_fn = cosine_lr(total_steps, warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_fn)
        print(f"Rank {rank}: Optimizer and scheduler created")
        if rank == 0:
            print(f"Base learning rate set to {base_lr:.6f}")
            print(f"Warmup epochs: {warmup_epochs} (warmup steps: {warmup_steps})")

        start_epoch = 1
        if resume_from_checkpoint:
            checkpoint_info: list[str | None] = [None]
            if rank == 0:
                latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
                if latest_checkpoint:
                    print(f"Found checkpoint for resume: {latest_checkpoint}")
                else:
                    print("No checkpoint found, starting DDP training from scratch")
                checkpoint_info[0] = latest_checkpoint

            dist.broadcast_object_list(checkpoint_info, src=0)
            latest_checkpoint = checkpoint_info[0]

            if latest_checkpoint:
                try:
                    last_epoch, _ = load_checkpoint(
                        model.module,
                        optimizer,
                        latest_checkpoint,
                        scheduler=scheduler,
                    )
                    start_epoch = last_epoch + 1
                    print(f"Rank {rank}: Resuming from epoch {start_epoch}")
                except Exception as e:
                    if rank == 0:
                        print(f"Failed to load checkpoint '{latest_checkpoint}': {e}")
                        print("Starting training from scratch")
                    start_epoch = 1

        start_epoch_list = [start_epoch]
        dist.broadcast_object_list(start_epoch_list, src=0)
        start_epoch = int(start_epoch_list[0])

        if start_epoch > epochs_total:
            if rank == 0:
                print(f"All {epochs_total} epochs already completed according to checkpoints.")
            return

        # Training loop
        model.train()  # type: ignore
        for epoch in range(start_epoch, epochs_total + 1):
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
                writer,
                scheduler,
            )

            # Launch S3 upload on rank 0 after each epoch (if enabled)
            if rank == 0 and enable_s3:
                launch_s3_upload_for_epoch(
                    epoch, checkpoint_dir, run_id, log_dir, s3_upload_processes
                )

            if rank == 0:
                cleanup_process = mp.Process(
                    target=clean_up_old_checkpoints,
                    args=(checkpoint_dir, checkpoints_to_keep),
                )
                cleanup_process.start()
                cleanup_processes.append((cleanup_process, f"cleanup after epoch {epoch}"))

    finally:
        if writer is not None:
            writer.close()  # type: ignore[no-untyped-call]

        # Report status of background processes but don't wait for them
        # S3 uploads and cleanup will continue in the background
        if rank == 0:
            if enable_s3:
                alive_s3 = [item for item in s3_upload_processes if item[0].is_alive()]
                if alive_s3:
                    print(f"Rank 0: {len(alive_s3)} S3 upload process(es) still running:")
                    for process, description in alive_s3:
                        print(f"  PID {process.pid}: {description}")

            alive_cleanup = [item for item in cleanup_processes if item[0].is_alive()]
            if alive_cleanup:
                print(
                    f"Rank 0: {len(alive_cleanup)} cleanup process(es) still running in background:"
                )
                for process, description in alive_cleanup:
                    print(f"  PID {process.pid}: {description}")

        # Synchronize all ranks before exit to ensure all training is complete
        # Background processes (S3 uploads, cleanup) will continue independently
        if dist.is_initialized():
            print(f"Rank {rank}: Synchronizing with other ranks before exit...")
            try:
                dist.barrier(device_ids=[rank])
                print(f"Rank {rank}: All ranks synchronized, training complete")
            except Exception as e:
                print(f"Rank {rank}: Barrier failed (another rank may have crashed): {e}")

        print(f"Rank {rank}: Training completed successfully, exiting...")


def launch_ddp_training(
    world_size: int | None = None,
    enable_s3: bool = True,
    resume_from_checkpoint: bool = True,
) -> None:
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

    # Find a free port for this training run
    free_port = find_free_port()
    os.environ["MASTER_PORT"] = str(free_port)
    print(f"Launching DDP training with {world_size} processes on port {free_port}")

    # Spawn training processes
    mp.spawn(  # type: ignore
        train_ddp_worker,
        args=(world_size, run_id, enable_s3, resume_from_checkpoint),
        nprocs=world_size,
        join=True,
    )


# Example usage
if __name__ == "__main__":
    # Launch DDP training with dummy data
    launch_ddp_training()
