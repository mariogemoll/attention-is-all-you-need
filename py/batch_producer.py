from __future__ import annotations

import queue
import time
from os import path
from typing import Literal, TypedDict

import torch
import torch.multiprocessing as mp

from batching import EpochBatches
from data import get_tensors, open_buckets
from per_process_logs import redirect_stdio


class InitMessage(TypedDict):
    type: Literal["start"]
    num_batches: int


class BatchMessage(TypedDict):
    type: Literal["batch"]
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


type DataQueueMessage = InitMessage | BatchMessage


def batch_producer(
    target_num_tokens_per_batch: int,
    dataset_base_path: str,
    num_procs: int,
    proc_id: int,
    device_id: str,
    data_queue: mp.Queue[DataQueueMessage],
    term_queue: mp.Queue[None],
    rng_seed: int,
    log_dir: str,
    epoch: int,
) -> None:
    redirect_stdio(path.join(log_dir, f"batch_producer_epoch_{epoch}_proc_{proc_id}.log"))

    device = torch.device(device_id)

    try:
        with open_buckets(dataset_base_path) as dataset:
            train_batches = EpochBatches(
                num_procs,
                proc_id,
                dataset.bucket_index_file,
                target_num_tokens_per_batch,
                rng_seed,
                True,
            )
            data_queue.put({"type": "start", "num_batches": len(train_batches)})
            for batch_id, entry_ids in train_batches:
                seq_len = (batch_id + 1) * dataset.step_size
                enc_input, dec_input, dec_target = get_tensors(
                    dataset.index_file,
                    dataset.data_file,
                    dataset.data_file_size,
                    seq_len,
                    entry_ids,
                )
                enc_input = enc_input.to(device)
                dec_input = dec_input.to(device)
                dec_target = dec_target.to(device)
                data_queue.put({"type": "batch", "data": (enc_input, dec_input, dec_target)})
        print("Done generating batches. Waiting for term signal")

        # Wait for termination signal with timeout and warning
        timeout_seconds = 3  # Adjust timeout as needed
        term_signal_wait_start = time.time()
        while True:
            try:
                term_queue.get(timeout=timeout_seconds)
                print("Received term signal. Exiting")
                break
            except queue.Empty:
                print(
                    "Warning: No termination signal received after "
                    f"{time.time() - term_signal_wait_start} seconds. Still waiting..."
                )
                continue
    except Exception as e:
        print(f"Batch producer error: {e}")
        raise
    finally:
        # Ensure CUDA context is properly cleaned up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all CUDA operations are complete
            torch.cuda.synchronize()
