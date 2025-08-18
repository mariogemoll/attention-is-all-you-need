import time
import torch
import torch.multiprocessing as mp
from queue import Queue
from threading import Thread
from model import Transformer
from params import max_seq_len
from tqdm import tqdm, trange
from util import get_device


def cpu_copy_worker(gpu_queue, cpu_queue):
    """Worker process that copies tensors from GPU to CPU"""
    while True:
        try:
            tensor = gpu_queue.get(timeout=1)
            if tensor is None:  # Sentinel to stop
                break
            cpu_tensor = tensor.cpu()
            cpu_queue.put(cpu_tensor.shape)
        except:
            break


def benchmark_encoder():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    print(f"Using device: {device}")
    
    model = Transformer().to(device)
    model.eval()

    # num_tokens = 1048576 * 16
    num_tokens = 1048576
    seq_len = 16
    # batch_size = 65536
    batch_size = 16384
    print('batch size', batch_size)
    tokens_per_batch = seq_len * batch_size
    print('tokens per batch', tokens_per_batch)
    num_batches = int(num_tokens // tokens_per_batch)
    print('num batches', num_batches)
    
    # Set up multiprocessing queues
    mp.set_start_method('spawn', force=True)
    gpu_queue = mp.Queue(maxsize=10)  # Buffer a few tensors
    cpu_queue = mp.Queue()
    
    # Start CPU copy worker process
    copy_process = mp.Process(target=cpu_copy_worker, args=(gpu_queue, cpu_queue))
    copy_process.start()
    
    with torch.no_grad():
        start_time = time.time()

        for i in trange(num_batches, desc="Benchmarking"):
            src_tensor = torch.randint(0, 32768, (batch_size, seq_len), device=device)
            encoded = model.encode(src_tensor)
            
            # Send to CPU copy process (non-blocking)
            try:
                gpu_queue.put(encoded, block=False)
            except:
                pass  # Skip if queue is full
            
            # Check for completed CPU copies
            try:
                while True:
                    shape = cpu_queue.get(block=False)
                    print(f"CPU copy completed: {shape}")
            except:
                pass  # No more completed copies

            
            # if (i + 1) % 100 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"Completed {i + 1}/1000 iterations in {elapsed:.2f}s")
        
        total_time = time.time() - start_time
    
    # Clean up
    gpu_queue.put(None)  # Sentinel to stop worker
    copy_process.join(timeout=5)
    if copy_process.is_alive():
        copy_process.terminate()
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per batch: {total_time/num_batches:.4f}s")
    print(f"Batches per second: {num_batches/total_time:.2f}")


if __name__ == "__main__":
    benchmark_encoder()