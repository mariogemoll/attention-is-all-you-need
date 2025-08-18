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
    bucket_idx = 0
    seq_len = (bucket_idx + 1) * 16
    # batch_size = 65536
   # batch_size = 16384
    # batch_size = 4096
    batch_size = 4
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
        
        # Keep track of which batch positions are occupied and by which sequences
        batch_positions = {}  # line_number -> batch_position
        position_to_line = {}  # batch_position -> line_number
        next_available_position = 0
        
        while True:
            while not all_inputs_consumed and (cache_size - len(sequences)) >= batch_size:
                print('Filling cache...')
                batch = list(islice(bucket_entries, batch_size))
                if len(batch) == 0:
                    all_inputs_consumed = True
                    break
                enc_input_list = []
                for _, src_tokens in batch:
                    # Pad the source tokens to seq_len
                    padded_src = src_tokens + [pad] * (seq_len - len(src_tokens))
                    enc_input_list.append(padded_src)
                enc_input_tensor = torch.tensor(enc_input_list, dtype=torch.long, device=device)
                print('input tensor shape', enc_input_tensor.shape)
                encoded = model.encode(enc_input_tensor)
                for i in range(len(batch)):
                    cache_loc = cache_free_list.get()
                    tokens_cache[cache_loc, 0, :] = enc_input_tensor[i]
                    tokens_cache[cache_loc, 1, :] = torch.tensor([sos] + [pad] * (seq_len - 1), dtype=torch.long)
                    decoded_seq_len_cache[cache_loc] = 1
                    encodings_cache[cache_loc, :, :] = encoded[i]
                    sequences[batch[i][0]] = cache_loc

            # Get current batch of sequences
            current_sequences = dict(islice(sequences.items(), batch_size))
            if len(current_sequences) == 0:
                print('No more sequences to decode')
                break

            # Update batch positions for new sequences
            for line_number in current_sequences:
                if line_number not in batch_positions:
                    # Find next available position or reuse a freed one
                    pos = None
                    # First try to find a freed position
                    for i in range(batch_size):
                        if i not in position_to_line:
                            pos = i
                            break
                    
                    # If no freed position, use next_available_position but constrain to batch_size
                    if pos is None:
                        while next_available_position in position_to_line and next_available_position < batch_size:
                            next_available_position += 1
                        if next_available_position < batch_size:
                            pos = next_available_position
                            next_available_position += 1
                        else:
                            # This should not happen if batch management is correct
                            raise RuntimeError(f"No available batch position for new sequence {line_number}")
                    
                    batch_positions[line_number] = pos
                    position_to_line[pos] = line_number
                    
                    # Update tensors for this new sequence
                    cache_loc = current_sequences[line_number]
                    enc_input[pos] = tokens_cache[cache_loc, 0, :]
                    memory[pos] = encodings_cache[cache_loc, :, :]
            
            # Update dec_input for all current sequences (as it changes each iteration)
            for line_number, cache_loc in current_sequences.items():
                pos = batch_positions[line_number]
                dec_input[pos] = tokens_cache[cache_loc, 1, :]
            

            # Create tensors with only the active sequences for this batch
            active_batch_size = len(current_sequences)
            active_enc_input = torch.empty((active_batch_size, seq_len), dtype=torch.long, device=device)
            active_memory = torch.empty((active_batch_size, seq_len, d_model), device=device)
            active_dec_input = torch.empty((active_batch_size, seq_len), dtype=torch.long, device=device)
            
            batch_pos_to_active_idx = {}
            for active_idx, (line_number, cache_loc) in enumerate(current_sequences.items()):
                batch_pos = batch_positions[line_number]
                batch_pos_to_active_idx[batch_pos] = active_idx
                active_enc_input[active_idx] = enc_input[batch_pos]
                active_memory[active_idx] = memory[batch_pos]
                active_dec_input[active_idx] = dec_input[batch_pos]
            
            result = model.decode(active_enc_input, active_memory, active_dec_input)


            # argmaxed = result.argmax(dim=-1)
            # print('result', argmaxed)
            # time.sleep(3)

            # Go through the result
            sequences_to_remove = []

            for active_idx, (line_number, cache_loc) in enumerate(current_sequences.items()):
                batch_pos = batch_positions[line_number]
                # Look at the newly generated token (at the current length of target tokens)
                current_tgt_len = decoded_seq_len_cache[cache_loc]

                new_token = int(result[active_idx][current_tgt_len - 1].argmax().item())


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
                    sequences_to_remove.append(line_number)
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
                    sequences_to_remove.append(line_number)
                    num_sequences_completed += 1
                    pbar.update(1)
                else:
                    # Otherwise, we just add the new token to the list
                    tokens_cache[cache_loc, 1, current_tgt_len] = new_token
                    decoded_seq_len_cache[cache_loc] += 1
            
            # Clean up batch positions for removed sequences
            for line_number in sequences_to_remove:
                if line_number in batch_positions:
                    pos = batch_positions[line_number]
                    del batch_positions[line_number]
                    del position_to_line[pos]


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
