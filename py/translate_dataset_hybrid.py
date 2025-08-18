import sys
from typing import Dict, Iterator, Tuple
import torch
from tqdm import tqdm
from batching import BucketEntries
from data import open_buckets
from model import Transformer
from params import eos, pad, sos, target_num_tokens_per_batch
from util import get_device

try:
    import translate_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("C++ extension not available, falling back to Python implementation")

class HybridTranslationEngine:
    def __init__(self, model_path: str, use_cpp: bool = True):
        self.device = get_device()
        self.model = Transformer()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.use_cpp = use_cpp and CPP_AVAILABLE
        
        if self.use_cpp:
            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            self.cpp_engine = translate_cpp.TranslationEngine(
                device_str, target_num_tokens_per_batch, pad, sos, eos
            )
            print("Using C++ acceleration")
        else:
            print("Using Python implementation")
    
    def translate_dataset(self, input_path: str, output_path: str):
        if self.use_cpp:
            self._translate_with_cpp(input_path, output_path)
        else:
            self._translate_with_python(input_path, output_path)
    
    def _translate_with_cpp(self, input_path: str, output_path: str):
        self.cpp_engine.initialize_output(output_path)
        
        with open_buckets(input_path) as dataset, torch.no_grad():
            for bucket_idx in range(dataset.num_buckets):
                bucket_entries = BucketEntries(dataset, bucket_id=bucket_idx)
                
                # Convert to format expected by C++
                bucket_data = []
                for line_num, src_tokens in bucket_entries:
                    bucket_data.append((line_num, src_tokens))
                
                step_size = dataset.step_size
                
                # Process with C++ engine
                self.cpp_engine.process_bucket(
                    self.model, bucket_data, bucket_idx, step_size
                )
    
    def _translate_with_python(self, input_path: str, output_path: str):
        # Fallback to original Python implementation
        from translate_dataset import main as python_main
        original_argv = sys.argv
        try:
            sys.argv = ['translate_dataset.py', sys.argv[1], input_path, output_path]
            python_main()
        finally:
            sys.argv = original_argv

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <model_path> <input_path> <output_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_path = sys.argv[2] 
    output_path = sys.argv[3]
    
    engine = HybridTranslationEngine(model_path)
    engine.translate_dataset(input_path, output_path)

if __name__ == "__main__":
    main()