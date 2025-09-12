vocab_size = 32768
max_seq_len = 176


pad = 0
sos = 1
eos = 2

num_model_dims = 512
num_heads = 8
num_layers = 6

# num_model_dims = 64
# num_heads = 2
# num_layers = 2

num_epochs = 10

seq_len_step_size = 16

target_num_tokens_per_batch = 20000

# Logging configuration
log_base_path = "../logs"
