import torch
import torch.multiprocessing as mp

from batch_producer import DataQueueMessage, batch_producer
from buckets import open_buckets
from evaluation import evaluate
from params import checkpoints_to_keep, target_num_tokens_per_batch
from training import clean_up_old_checkpoints, init, save_checkpoint, train_one_epoch


def main() -> None:
    checkpoint_dir = "checkpoints"
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)  # type: ignore
    torch.autograd.profiler.emit_nvtx(False)  # type: ignore
    torch.set_float32_matmul_precision("high")

    mp.set_start_method("spawn", force=True)

    device_id = "cuda:0"
    device = torch.device(device_id)

    model, optimizer, criterion, writer, start_epoch = init(
        device, checkpoint_dir, resume_from_checkpoint=True
    )

    model.compile(dynamic=True)  # type: ignore

    total_epochs = 10

    with open_buckets("../4_tokens/newstest2013") as valset:
        for epoch in range(start_epoch, total_epochs + 1):

            data_queue: mp.Queue[DataQueueMessage] = mp.Queue(maxsize=10)
            term_queue: mp.Queue[None] = mp.Queue()

            rng_seed = 42 + epoch
            # Start batch_producer as separate process
            batch_producer_proc = mp.Process(
                target=batch_producer,
                args=(
                    target_num_tokens_per_batch,
                    "../4_tokens/newstest2013",
                    1,
                    0,
                    device_id,
                    data_queue,
                    term_queue,
                    rng_seed,
                ),
            )
            batch_producer_proc.start()

            # Train for one epoch
            train_one_epoch(model, optimizer, criterion, writer, data_queue, epoch)

            term_queue.put(None)

            batch_producer_proc.join()

            # Evaluate and get validation loss
            val_loss = evaluate(device, valset, model, criterion, writer, epoch)

            # Save checkpoint after each epoch (includes model weights)
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)

            # Clean up old checkpoints (keep only the last N)
            clean_up_old_checkpoints(checkpoint_dir, checkpoints_to_keep)

    # Close TensorBoard writer
    writer.close()  # type: ignore


if __name__ == "__main__":
    main()
