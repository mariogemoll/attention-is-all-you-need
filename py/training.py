import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

from batching import EpochBatches
from data import BucketedDataset, get_tensors
from model import Transformer
from params import pad
from util import get_device


def init() -> tuple[torch.device, Transformer, AdamW, CrossEntropyLoss]:
    device = get_device()
    model = Transformer().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-10)
    criterion = CrossEntropyLoss(ignore_index=pad)
    return device, model, optimizer, criterion


def train_one_epoch(
    device: torch.device,
    trainset: BucketedDataset,
    model: Transformer,
    optimizer: AdamW,
    criterion: CrossEntropyLoss,
    epoch: int,
) -> None:
    model.train()
    train_batches = EpochBatches(
        trainset.bucket_index_file,
        target_tokens_per_batch=25000,
        shuffle_within_buckets=True,
        shuffle_batches=True,
        random_seed=42,
        full_batches_only=True,
    )

    pbar = tqdm(train_batches, desc=f"Epoch {epoch}")
    for batch_id, entry_ids in pbar:
        seq_len = (batch_id + 1) * trainset.step_size
        enc_input, dec_input, dec_target = get_tensors(
            trainset.index_file, trainset.data_file, trainset.data_file_size, seq_len, entry_ids
        )
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        dec_target = dec_target.to(device)
        optimizer.zero_grad()
        memory = model.encode(enc_input)
        out = model.decode(enc_input, memory, dec_input)
        loss = criterion(out.view(-1, out.size(-1)), dec_target.view(-1))
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        optimizer.step()


def evaluate(
    device: torch.device,
    valset: BucketedDataset,
    model: Transformer,
    criterion: CrossEntropyLoss,
    epoch: int,
) -> None:
    model.eval()
    val_batches = EpochBatches(
        valset.bucket_index_file,
        target_tokens_per_batch=25000,
        shuffle_within_buckets=False,
        shuffle_batches=False,
        random_seed=42,
        full_batches_only=True,
    )
    with torch.no_grad():
        losses = torch.zeros(len(val_batches))
        pbar = tqdm(val_batches, desc=f"Evaluating Epoch {epoch}")
        for i, (batch_id, entry_ids) in enumerate(pbar):
            seq_len = (batch_id + 1) * valset.step_size
            enc_input, dec_input, dec_target = get_tensors(
                valset.index_file, valset.data_file, valset.data_file_size, seq_len, entry_ids
            )
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            memory = model.encode(enc_input)
            out = model.decode(enc_input, memory, dec_input)
            loss = criterion(out.view(-1, out.size(-1)), dec_target.view(-1))
            losses[i] = loss
            pbar.set_postfix({"loss": loss.item()})
    epoch_loss = losses.mean().item()
    print(f"Validation loss for epoch {epoch}: {epoch_loss}")
