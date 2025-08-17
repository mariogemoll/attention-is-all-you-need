import time

import torch
import torch.multiprocessing as mp
from multiprocessing.connection import wait

from model import Transformer
from util import get_device

def run_processor(foo, model_weights_path, mgr_to_enc_queue: mp.Queue, enc_to_mgr_queue: mp.Queue, mgr_to_dec_queue: mp.Queue,
                  dec_to_mgr_queue: mp.Queue):
    print(f"processor: foo {foo}")
    # device = get_device()
    device = 'cpu'
    model = Transformer()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():

        readers = [mgr_to_enc_queue._reader, mgr_to_dec_queue._reader]
        while True:
            start = time.time()
            ready = wait(readers)
            wait_time = time.time() - start
            print(f'Waited {wait_time * 1000} ms')
            # Enc has priority
            if mgr_to_enc_queue._reader in ready:
                enc_input = mgr_to_enc_queue.get_nowait()
                enc_input_tensor = torch.tensor(enc_input, dtype=torch.long, device=device)
                encoded = model.encode(enc_input_tensor)
                print('processor: encoding done. sending enc result')
                enc_to_mgr_queue.put(encoded)
                print('processor: sent enc result')
            else:
                enc_input_tensor, memory_tensor, dec_input_tensor = mgr_to_dec_queue.get()
                print('processor: got dec input')
                result = model.decode(enc_input_tensor, memory_tensor, dec_input_tensor)
                print('processor: decoding done. sending dec result')
                dec_to_mgr_queue.put(result)
                print('processor: sent dec result')
    