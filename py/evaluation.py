import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from batching import EpochBatches
from buckets import BucketedDataset
from model import Transformer
from params import target_num_tokens_per_batch
from tensors import get_tensors


def evaluate(
    device: torch.device,
    valset: BucketedDataset,
    model: Transformer,
    criterion: CrossEntropyLoss,
    writer: SummaryWriter,
    epoch: int,
) -> float:
    model.eval()
    val_batches = EpochBatches(
        1,
        0,
        valset.bucket_index_file,
        target_num_tokens_per_batch,
        rng_seed=42,
        full_batches_only=True,
    )
    with torch.no_grad():
        losses = torch.zeros(len(val_batches))
        pbar = tqdm(val_batches, desc=f"Evaluating Epoch {epoch}")
        for i, (batch_id, entry_ids) in enumerate(pbar):
            seq_len = (batch_id + 1) * valset.step_size
            enc_input, dec_input, dec_target = get_tensors(seq_len, valset.dataset, entry_ids)
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            memory = model.encode(enc_input)
            out = model.decode(enc_input, memory, dec_input)
            loss = criterion(out.view(-1, out.size(-1)), dec_target.view(-1))
            losses[i] = loss
            pbar.set_postfix({"loss": loss.item()})
    epoch_loss = losses.mean().item()
    writer.add_scalar("loss/epoch/val", epoch_loss, epoch)  # type: ignore
    print(f"Validation loss for epoch {epoch}: {epoch_loss}")
    return epoch_loss
