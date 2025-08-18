import sys
import time
import torch
import torch.multiprocessing as mp
from queue import Queue
from threading import Thread
from model import Transformer
from params import max_seq_len
from tqdm import tqdm, trange
from util import get_device
from data import open_buckets
from batching import BucketEntries
from itertools import islice
from params import pad, sos, eos
import tokenizers

torch.set_printoptions(linewidth=200)  # widen the line before breaking


def cpu_copy_worker(gpu_queue, cpu_queue):
    """Worker process that copies tensors from GPU to CPU"""
    while True:
        try:
            tensor = gpu_queue.get(timeout=1)
            if tensor is None:  # Sentinel to stop
                break
            cpu_tensor = tensor.cpu()
            del tensor
            cpu_queue.put(cpu_tensor.shape)
        except:
            break


def run():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    print(f"Using device: {device}")
    
    model = Transformer()
    model.to(device)
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
    model.eval()

    # num_tokens = 1048576 * 16
    # num_tokens = 1048576
    bucket_idx = 8
    seq_len = (bucket_idx + 1) * 16
    # batch_size = 65536
   # batch_size = 16384
    batch_size = 4096
    # batch_size = 4
    # batch_size = 8192
    # tokens_per_batch = seq_len * batch_size
    # num_batches = int(num_tokens // tokens_per_batch)

    input_path = '../4_tokens/val'

    tokenizer = tokenizers.Tokenizer.from_file(sys.argv[2])

    d_model = 512
    enc_input = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    memory = torch.empty((batch_size, seq_len, d_model), device=device)
    dec_input = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    with open_buckets(input_path) as dataset, torch.no_grad():
        bucket_entries = BucketEntries(dataset, bucket_id=bucket_idx)

        pbar = tqdm(total=len(bucket_entries))
        # start_time = time.time()

        multiplier = 2
        cache_size = batch_size * multiplier

        tokens_cache = torch.empty((cache_size, 2, seq_len), dtype=torch.long, device=device)
        encodings_cache = torch.empty((cache_size, seq_len, d_model), device=device)
        decoded_seq_len_cache = torch.empty((cache_size,), dtype=torch.long, device=device)
        cache_free_list = Queue()
        for i in range(cache_size):
            cache_free_list.put(i)

        num_sequences_completed = 0
        spillovers = []
        sequences = {}

        bucket_entries = iter(bucket_entries)
        all_inputs_consumed = False
        while True:
            while not all_inputs_consumed and (cache_size - len(sequences)) >= batch_size:
                print('Filling cache...')
                batch = list(islice(bucket_entries, batch_size))
                if len(batch) == 0:
                    all_inputs_consumed = True
                    break
                enc_input = []
                for _, src_tokens in batch:
                    # Pad the source tokens to seq_len
                    padded_src = src_tokens + [pad] * (seq_len - len(src_tokens))
                    enc_input.append(padded_src)
                enc_input_tensor = torch.tensor(enc_input, dtype=torch.long, device=device)
                print('input tensor shape', enc_input_tensor.shape)
                encoded = model.encode(enc_input_tensor)
                for i in range(len(batch)):
                    cache_loc = cache_free_list.get()
                    tokens_cache[cache_loc, 0, :] = enc_input_tensor[i]
                    tokens_cache[cache_loc, 1, :] = torch.tensor([sos] + [pad] * (seq_len - 1), dtype=torch.long)
                    decoded_seq_len_cache[cache_loc] = 1
                    encodings_cache[cache_loc, :, :] = encoded[i]
                    sequences[batch[i][0]] = cache_loc

            batch = dict(islice(sequences.items(), batch_size))
            if len(batch) == 0:
                print('No more sequences to decode')
                break


            cache_locs = list(batch.values())
            enc_input = tokens_cache[cache_locs, 0, :]
            memory = encodings_cache[cache_locs, :, :]
            dec_input = tokens_cache[cache_locs, 1, :]

            # print('enc_input', enc_input)
            # print('dec_input', dec_input)
            result = model.decode(enc_input, memory, dec_input)


            # argmaxed = result.argmax(dim=-1)
            # print('result', argmaxed)
            # time.sleep(3)

            # Go through the result

            for i, (line_number, cache_loc) in enumerate(batch.items()):
                # Look at the newly generated token (at the current length of target tokens)
                current_tgt_len = decoded_seq_len_cache[cache_loc]

                new_token = int(result[i][current_tgt_len - 1].argmax().item())


                if new_token == pad:
                    raise ValueError("Generated padding token")

                if new_token == eos:
                    # If the sequence ends with eos, it's completely translated. Store the
                    # result and remove the sequence from sequences
                    final_sequence = tokens_cache[cache_loc, 1, 1:current_tgt_len].tolist()
                    # store_output(
                    #     output_index_file, output_data_file, line_number - 1, final_sequence
                    # )
                    # print('src sequence', tokenizer.decode(tokens_cache[cache_loc, 0, :].tolist()))
                    # print('final sequence', tokenizer.decode(final_sequence), spillovers)
                    sequences.pop(line_number, None)
                    cache_free_list.put(cache_loc)
                    num_sequences_completed += 1
                    pbar.update(1)
                elif current_tgt_len >= seq_len - 1:
                    # We've exhausted the sequence length for this bucket (and we haven't
                    # generated eos yet), so we need to move this sequence into the next bucket
                    enc_input_tokens = tokens_cache[cache_loc, 0, :].tolist()
                    dec_input_tokens = tokens_cache[cache_loc, 1, 1:].tolist() + [new_token]
                    # info.tgt_tokens.append(new_token)
                    # spillover_sequences[line_number] = final_sequence
                    # print('sequence too long, truncating:', final_sequence)
                    # raise NotImplementedError("Spillover sequences not implemented")
                    # print('spillover:', final_sequence)
                    spillovers.append((line_number, enc_input_tokens, dec_input_tokens))
                    sequences.pop(line_number, None)
                    cache_free_list.put(cache_loc)
                    num_sequences_completed += 1
                    pbar.update(1)
                else:
                    # Otherwise, we just add the new token to the list
                    tokens_cache[cache_loc, 1, current_tgt_len] = new_token
                    decoded_seq_len_cache[cache_loc] += 1


        # print('total entries in bucket', len(bucket_entries))
        # # Get the first entry from bucketentries generator
        # first_entry = next(iter(bucket_entries))
        # print(first_entry)




    #     for i in trange(num_batches, desc="Benchmarking"):
    #         src_tensor = torch.randint(0, 32768, (batch_size, seq_len), device=device)
    #         encoded = model.encode(src_tensor)
            
    #         gpu_queue.put(encoded)
    #         # , block=False)
            
    #         # # Check for completed CPU copies
    #         # try:
    #         #     while True:
    #         #         shape = cpu_queue.get(block=False)
    #         #         print(f"CPU copy completed: {shape}")
    #         # except:
    #         #     pass  # No more completed copies

            
    #         # if (i + 1) % 100 == 0:
    #         #     elapsed = time.time() - start_time
    #         #     print(f"Completed {i + 1}/1000 iterations in {elapsed:.2f}s")
        
    #     total_time = time.time() - start_time

    # # Read the results
    # while not cpu_queue.empty():
    #     shape = cpu_queue.get()
    #     print(f"CPU copy completed: {shape}")

    # # Clean up
    # gpu_queue.put(None)  # Sentinel to stop worker
    # copy_process.join(timeout=5)
    # if copy_process.is_alive():
    #     copy_process.terminate()
    
    # print(f"\nTotal time: {total_time:.2f}s")
    # print(f"Average time per batch: {total_time/num_batches:.4f}s")
    # print(f"Batches per second: {num_batches/total_time:.2f}")


if __name__ == "__main__":
    run()
