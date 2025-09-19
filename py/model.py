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
            enable_nested_tensor=False,
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

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize TransformerEncoder layers
        for layer in self.encoder.layers:
            self._init_transformer_layer(layer)

        # Initialize TransformerDecoder layers
        for layer in self.decoder.layers:
            self._init_transformer_layer(layer)

        # Initialize embeddings with normal distribution
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=0.1)

        # Initialize the final linear layer
        nn.init.xavier_uniform_(self.to_token_map.weight)

    def _init_transformer_layer(self, layer: nn.Module) -> None:
        # Initialize multi-head attention
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, nn.MultiheadAttention):
            self._init_attention(layer.self_attn)
        if hasattr(layer, "multihead_attn") and isinstance(
            layer.multihead_attn, nn.MultiheadAttention
        ):
            self._init_attention(layer.multihead_attn)

        # Initialize feed-forward network
        if hasattr(layer, "linear1") and isinstance(layer.linear1, nn.Linear):
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
        if hasattr(layer, "linear2") and isinstance(layer.linear2, nn.Linear):
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

        # Initialize layer norms
        if hasattr(layer, "norm1") and isinstance(layer.norm1, nn.LayerNorm):
            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
        if hasattr(layer, "norm2") and isinstance(layer.norm2, nn.LayerNorm):
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)
        if hasattr(layer, "norm3") and isinstance(layer.norm3, nn.LayerNorm):  # For decoder layers
            nn.init.ones_(layer.norm3.weight)
            nn.init.zeros_(layer.norm3.bias)

    def _init_attention(self, attn: nn.MultiheadAttention) -> None:
        # Initialize attention weights
        nn.init.xavier_uniform_(attn.in_proj_weight)
        nn.init.zeros_(attn.in_proj_bias)
        nn.init.xavier_uniform_(attn.out_proj.weight)
        nn.init.zeros_(attn.out_proj.bias)

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
