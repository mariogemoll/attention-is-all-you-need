vocab_size = 32768
max_seq_len = 176

pad = 0
sos = 1
eos = 2

num_model_dims = 512
num_heads = 8
num_layers = 6

num_epochs = 10

seq_len_step_size = 16

target_num_tokens_per_batch = 20000

log_base_path = "../logs"

max_parallelism = 16

checkpoints_to_keep = 10

train_dataset_path = "../4_tokens/train"

# Values from the "Attention Is All You Need" paper
aiayn_tokens_per_step = 25_000
aiayn_warmup_steps = 4_000
target_num_processed_tokens = 100000 * aiayn_tokens_per_step
