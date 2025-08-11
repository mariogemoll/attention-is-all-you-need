import math

import torch
from torch import nn

from params import max_seq_len, num_heads, num_layers, num_model_dims, pad, vocab_size

embedding_multiplier = torch.sqrt(torch.tensor(num_model_dims, dtype=torch.float32))


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()

        # Create a long enough tensor for the positional encodings
        pe = torch.zeros(max_len, d_model)

        # Position indices (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Dividing by powers of 10000 based on even/odd index of d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices (2i) and cosine to odd indices (2i+1) in the embedding dim.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        # Register as buffer so it doesn't get updated during training
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be of shape (batch_size, sequence_length, d_model)
        # Add positional encoding to the input embedding
        x = x + self.pe[:, : x.size(1), :]
        return x


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_model_dims, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=num_model_dims, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )
        self.src_embedding = nn.Embedding(vocab_size, num_model_dims, padding_idx=pad)
        self.tgt_embedding = nn.Embedding(vocab_size, num_model_dims, padding_idx=pad)

        self.positional_encoding = PositionalEncoding(num_model_dims, max_len=max_seq_len)

        self.norm = nn.LayerNorm(num_model_dims)
        self.to_token_map = nn.Linear(num_model_dims, vocab_size, bias=False)

        # Weight tying
        self.to_token_map.weight = self.tgt_embedding.weight

        # TODO Initialize weights

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src_padding_mask = src == pad
        x: torch.Tensor = self.src_embedding(src) * torch.sqrt(
            torch.tensor(num_model_dims, dtype=torch.float32)
        )
        x = self.positional_encoding(x)
        x = self.norm(x)
        x = self.encoder(x, src_key_padding_mask=src_padding_mask)
        return x

    def decode(self, src: torch.Tensor, memory: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_padding_mask = src == pad
        tgt_padding_mask = tgt == pad
        y = self.tgt_embedding(tgt) * embedding_multiplier
        y = self.positional_encoding(y)
        # Make causal mask
        tgt_mask = torch.triu(
            torch.ones((tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1
        ).bool()
        out: torch.Tensor = self.decoder(
            tgt=y,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        out = self.norm(out)
        out = self.to_token_map(out)
        return out
