import sys
import torch.multiprocessing as mp
from processor import run_processor
from manager import run_manager

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <model_path> <input_path> <output_path>")
        sys.exit(1)
    
    model_path, input_path, output_path = sys.argv[1:]
    mgr_to_enc_queue = mp.Queue(maxsize=1)
    enc_to_mgr_queue = mp.Queue(maxsize=1)
    mgr_to_dec_queue = mp.Queue(maxsize=1)
    dec_to_mgr_queue = mp.Queue(maxsize=1)

    proc = mp.spawn(
        run_processor,
        args=(model_path, mgr_to_enc_queue, enc_to_mgr_queue, mgr_to_dec_queue, dec_to_mgr_queue),
        join=False
    )

    # mgr = mp.spawn(
    #     run_manager,
    #     args=(input_path, output_path, mgr_to_enc_queue, enc_to_mgr_queue, mgr_to_dec_queue, dec_to_mgr_queue),
    #     join=False
    # )
    run_manager('bla', input_path, output_path, mgr_to_enc_queue, enc_to_mgr_queue,
                mgr_to_dec_queue, dec_to_mgr_queue)

    # run_processor('foo', model_path, mgr_to_enc_queue, enc_to_mgr_queue, mgr_to_dec_queue, dec_to_mgr_queue)
    proc.join()
    # mgr.join()

if __name__ == '__main__':
    main()