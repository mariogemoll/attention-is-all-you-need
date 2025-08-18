import sys
from itertools import islice
from typing import BinaryIO, Dict, Iterator, List, Tuple
import torch
from tqdm import tqdm
import numpy as np

from batching import BucketEntries
from data import open_buckets
from model import Transformer
from params import eos, pad, sos, target_num_tokens_per_batch
from util import get_device


class OptimizedSequenceInfo:
    __slots__ = ['src_tokens', 'src_encoding', 'tgt_tokens', 'line_number']
    
    def __init__(self, src_tokens: List[int], src_encoding: torch.Tensor, 
                 tgt_tokens: List[int], line_number: int):
        self.src_tokens = src_tokens
        self.src_encoding = src_encoding
        self.tgt_tokens = tgt_tokens
        self.line_number = line_number


class OptimizedBinaryWriter:
    def __init__(self, output_path: str):
        self.data_file = open(output_path + ".bin", "wb")
        self.index_file = open(output_path + ".idx", "wb")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data_file.close()
        self.index_file.close()
    
    def store_output(self, entry_idx: int, data: List[int]) -> None:
        if len(data) == 0:
            pos = 0
        else:
            pos = self.data_file.tell()
            # Use numpy for faster conversion
            data_array = np.array(data, dtype=np.int16)
            self.data_file.write(data_array.tobytes())
        
        # Write index entry
        idx_file_pos = entry_idx * 5
        self.index_file.seek(idx_file_pos)
        self.index_file.write(pos.to_bytes(4, byteorder="little"))
        self.index_file.write(len(data).to_bytes(1, byteorder="little"))


class TensorCache:
    def __init__(self, device: torch.device, max_seq_len: int, d_model: int = 512):
        self.device = device
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Pre-allocate tensors for reuse
        self.enc_input_cache = torch.empty((256, max_seq_len), dtype=torch.long, device=device)
        self.dec_input_cache = torch.empty((256, max_seq_len), dtype=torch.long, device=device)
        self.memory_cache = torch.empty((256, max_seq_len, d_model), device=device)
        
    def get_tensors(self, batch_size: int, seq_len: int):
        return (
            self.enc_input_cache[:batch_size, :seq_len],
            self.dec_input_cache[:batch_size, :seq_len],
            self.memory_cache[:batch_size, :seq_len, :]
        )


class OptimizedTranslationEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.sequences: Dict[int, OptimizedSequenceInfo] = {}
        self.spillover_sequences: Dict[int, OptimizedSequenceInfo] = {}
        
    def add_src_sequences_batch(self, model: Transformer, seq_len: int, 
                               batch: List[Tuple[int, List[int]]], 
                               tensor_cache: TensorCache) -> int:
        if not batch:
            return 0
            
        batch_size = len(batch)
        enc_input, _, _ = tensor_cache.get_tensors(batch_size, seq_len)
        
        # Fill tensor directly without creating intermediate lists
        for i, (line_num, src_tokens) in enumerate(batch):
            src_len = min(len(src_tokens), seq_len)
            enc_input[i, :src_len] = torch.tensor(src_tokens[:src_len], dtype=torch.long)
            enc_input[i, src_len:] = pad
            
        encoded = model.encode(enc_input[:batch_size])
        
        # Create sequence infos
        for i, (line_num, src_tokens) in enumerate(batch):
            self.sequences[line_num] = OptimizedSequenceInfo(
                src_tokens, encoded[i].clone(), [sos], line_num
            )
            
        return batch_size

    def process_batch(self, model: Transformer, batch_sequences: List[OptimizedSequenceInfo],
                     seq_len: int, tensor_cache: TensorCache, writer: OptimizedBinaryWriter) -> Tuple[int, List[OptimizedSequenceInfo]]:
        if not batch_sequences:
            return 0, []
            
        batch_size = len(batch_sequences)
        enc_input, dec_input, memory = tensor_cache.get_tensors(batch_size, seq_len)
        
        # Fill tensors efficiently
        for i, seq_info in enumerate(batch_sequences):
            # Source tokens
            src_len = min(len(seq_info.src_tokens), seq_len)
            enc_input[i, :src_len] = torch.tensor(seq_info.src_tokens[:src_len], dtype=torch.long)
            enc_input[i, src_len:] = pad
            
            # Target tokens  
            tgt_len = min(len(seq_info.tgt_tokens), seq_len)
            dec_input[i, :tgt_len] = torch.tensor(seq_info.tgt_tokens[:tgt_len], dtype=torch.long)
            dec_input[i, tgt_len:] = pad
            
            # Memory (pre-computed encoding)
            memory[i] = seq_info.src_encoding[:seq_len]
            
        # Model forward pass
        result = model.decode(enc_input[:batch_size], memory[:batch_size], dec_input[:batch_size])
        
        # Process results
        completed = 0
        new_spillovers = []
        
        for i, seq_info in enumerate(batch_sequences):
            current_tgt_len = len(seq_info.tgt_tokens)
            new_token = int(result[i][current_tgt_len - 1].argmax().item())
            
            if new_token == pad:
                raise ValueError("Generated padding token")
                
            if new_token == eos:
                # Sequence complete
                final_sequence = seq_info.tgt_tokens[1:]  # Remove SOS
                writer.store_output(seq_info.line_number - 1, final_sequence)
                completed += 1
                
            elif current_tgt_len >= seq_len:
                # Move to spillover
                seq_info.tgt_tokens.append(new_token)
                new_spillovers.append(seq_info)
                completed += 1
                
            else:
                # Continue sequence
                seq_info.tgt_tokens.append(new_token)
                self.sequences[seq_info.line_number] = seq_info
                
        return completed, new_spillovers


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <model_path> <input_path> <output_path>")
        sys.exit(1)

    device = get_device()
    model = Transformer()
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
    model.to(device)
    model.eval()

    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    engine = OptimizedTranslationEngine(device)
    tensor_cache = TensorCache(device, 512)  # Max sequence length
    
    with open_buckets(input_path) as dataset, OptimizedBinaryWriter(output_path) as writer, torch.no_grad():
        for bucket_idx in range(dataset.num_buckets):
            bucket_entries = BucketEntries(dataset, bucket_id=bucket_idx)
            seq_len = (bucket_idx + 1) * dataset.step_size
            num_rows = max(4, (target_num_tokens_per_batch // seq_len) // 16 * 16)
            
            print(f"Processing bucket {bucket_idx} with seq_len={seq_len}, batch_size={num_rows}")
            
            total_completed = 0
            bucket_iterator = iter(bucket_entries)
            
            # Handle spillovers from previous bucket
            spillover_list = list(engine.spillover_sequences.values())
            engine.spillover_sequences.clear()
            
            pbar = tqdm(total=len(bucket_entries) + len(spillover_list), desc=f"Bucket {bucket_idx}")
            
            while True:
                # Get current sequences for processing
                current_sequences = list(engine.sequences.values())
                
                # Add spillovers if needed
                if spillover_list:
                    current_sequences.extend(spillover_list[:num_rows - len(current_sequences)])
                    spillover_list = spillover_list[num_rows - len(current_sequences):]
                
                # Fill with new sequences if needed
                if len(current_sequences) < num_rows:
                    new_batch = list(islice(bucket_iterator, num_rows - len(current_sequences)))
                    if new_batch:
                        engine.add_src_sequences_batch(model, seq_len, new_batch, tensor_cache)
                        current_sequences = list(engine.sequences.values())
                
                if not current_sequences:
                    break
                    
                # Process batch
                engine.sequences.clear()
                completed, new_spillovers = engine.process_batch(
                    model, current_sequences, seq_len, tensor_cache, writer
                )
                
                # Update spillovers
                for spillover in new_spillovers:
                    engine.spillover_sequences[spillover.line_number] = spillover
                
                total_completed += completed
                pbar.update(completed)
                
            pbar.close()
            print(f"Bucket {bucket_idx} completed: {total_completed} sequences")
            
        # Handle remaining spillovers
        for line_number, seq_info in engine.spillover_sequences.items():
            writer.store_output(line_number - 1, [])


if __name__ == "__main__":
    main()