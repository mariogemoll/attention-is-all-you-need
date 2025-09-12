import os
import tempfile
from typing import Generator

import pytest
import torch
import torch.multiprocessing as mp

from batch_producer import DataQueueMessage, batch_producer
from buckets import create_bucket_index
from dataset_test_helpers import make_toy_dataset


@pytest.fixture(scope="module")
def shared_dataset() -> Generator[str, None, None]:
    """Create a shared dataset for all tests to avoid repeated creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        # Create minimal test dataset
        entries = [
            ([1, 2], [3, 4]),
            ([5, 6], [7, 8]),
        ]
        make_toy_dataset(base, entries)
        create_bucket_index(base, step_size=2, max_length=8, num_processes=1)

        yield base


def test_batch_producer_basic(shared_dataset: str) -> None:
    """Test basic functionality of batch_producer."""
    # Create queues
    data_queue: mp.Queue[DataQueueMessage] = mp.Queue()
    term_queue: mp.Queue[None] = mp.Queue()

    # Start batch producer in a separate process
    process = mp.Process(
        target=batch_producer,
        args=(
            4,  # target_num_tokens_per_batch (smaller for faster test)
            shared_dataset,  # dataset_base_path
            1,  # num_procs
            0,  # proc_id
            "cpu",  # device_id
            data_queue,
            term_queue,
            42,  # rng_seed
        ),
    )

    try:
        process.start()

        # Get init message
        init_msg = data_queue.get(timeout=2)  # Reduced timeout
        assert init_msg["type"] == "start"
        assert isinstance(init_msg["num_batches"], int)
        assert init_msg["num_batches"] >= 0

        # Get batch messages
        batches_received = 0
        while batches_received < init_msg["num_batches"]:
            batch_msg = data_queue.get(timeout=2)  # Reduced timeout
            assert batch_msg["type"] == "batch"

            enc_input, dec_input, dec_target = batch_msg["data"]

            # Verify tensor types and device
            assert isinstance(enc_input, torch.Tensor)
            assert isinstance(dec_input, torch.Tensor)
            assert isinstance(dec_target, torch.Tensor)
            assert enc_input.device.type == "cpu"
            assert dec_input.device.type == "cpu"
            assert dec_target.device.type == "cpu"

            # Verify tensor shapes are compatible
            assert enc_input.shape == dec_input.shape == dec_target.shape
            assert len(enc_input.shape) == 2  # (batch_size, seq_len)

            batches_received += 1

        # Send termination signal
        term_queue.put(None)
        process.join(timeout=2)  # Reduced timeout
        assert process.exitcode == 0

    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_batch_producer_message_types(shared_dataset: str) -> None:
    """Test that batch_producer sends correct message types."""
    data_queue: mp.Queue[DataQueueMessage] = mp.Queue()
    term_queue: mp.Queue[None] = mp.Queue()

    process = mp.Process(
        target=batch_producer, args=(4, shared_dataset, 1, 0, "cpu", data_queue, term_queue, 42)
    )

    try:
        process.start()

        # First message should be InitMessage
        msg = data_queue.get(timeout=2)  # Reduced timeout
        assert msg["type"] == "start"
        assert "num_batches" in msg

        # Subsequent messages should be BatchMessage
        if msg["num_batches"] > 0:
            batch_msg = data_queue.get(timeout=2)  # Reduced timeout
            assert batch_msg["type"] == "batch"
            assert "data" in batch_msg
            assert len(batch_msg["data"]) == 3  # enc_input, dec_input, dec_target

        term_queue.put(None)
        process.join(timeout=2)  # Reduced timeout

    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_batch_producer_error_handling() -> None:
    """Test batch_producer error handling with invalid dataset path."""
    invalid_path = "/nonexistent/path"

    data_queue: mp.Queue[DataQueueMessage] = mp.Queue()
    term_queue: mp.Queue[None] = mp.Queue()

    process = mp.Process(
        target=batch_producer, args=(8, invalid_path, 1, 0, "cpu", data_queue, term_queue, 42)
    )

    try:
        process.start()
        process.join(timeout=2)  # Reduced timeout

        # Process should exit with non-zero code due to error
        assert process.exitcode != 0

    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_batch_producer_device_placement(shared_dataset: str) -> None:
    """Test that tensors are placed on the correct device."""
    data_queue: mp.Queue[DataQueueMessage] = mp.Queue()
    term_queue: mp.Queue[None] = mp.Queue()

    process = mp.Process(
        target=batch_producer, args=(4, shared_dataset, 1, 0, "cpu", data_queue, term_queue, 42)
    )

    try:
        process.start()

        # Get init message
        init_msg = data_queue.get(timeout=2)  # Reduced timeout
        assert init_msg["type"] == "start"

        # Get batch and verify device placement
        if init_msg["num_batches"] > 0:
            batch_msg = data_queue.get(timeout=2)  # Reduced timeout
            assert batch_msg["type"] == "batch"
            enc_input, dec_input, dec_target = batch_msg["data"]

            assert enc_input.device.type == "cpu"
            assert dec_input.device.type == "cpu"
            assert dec_target.device.type == "cpu"

        term_queue.put(None)
        process.join(timeout=2)  # Reduced timeout

    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_batch_producer_multiple_processes(shared_dataset: str) -> None:
    """Test batch_producer with multiple processes."""
    num_procs = 2
    processes = []
    data_queues = []
    term_queues = []

    try:
        # Start multiple batch producers
        for proc_id in range(num_procs):
            data_queue: mp.Queue[DataQueueMessage] = mp.Queue()
            term_queue: mp.Queue[None] = mp.Queue()
            data_queues.append(data_queue)
            term_queues.append(term_queue)

            process = mp.Process(
                target=batch_producer,
                args=(4, shared_dataset, num_procs, proc_id, "cpu", data_queue, term_queue, 42),
            )
            processes.append(process)
            process.start()

        # Collect results from all processes - just check init messages for speed
        total_batches = 0
        for proc_id in range(num_procs):
            init_msg = data_queues[proc_id].get(timeout=2)  # Reduced timeout
            assert init_msg["type"] == "start"
            total_batches += init_msg["num_batches"]

            # Only get first batch from each process to verify it works
            if init_msg["num_batches"] > 0:
                batch_msg = data_queues[proc_id].get(timeout=2)  # Reduced timeout
                assert batch_msg["type"] == "batch"

        # Send termination signals
        for term_queue in term_queues:
            term_queue.put(None)

        # Wait for all processes
        for process in processes:
            process.join(timeout=2)  # Reduced timeout
            assert process.exitcode == 0

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    # Set multiprocessing start method for testing
    mp.set_start_method("spawn", force=True)

    import pytest

    pytest.main([__file__, "-v"])
