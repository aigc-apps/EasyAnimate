import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import (PixArtAlphaTextProjection,
                                         TimestepEmbedding, Timesteps,
                                         get_timestep_embedding)
from einops import rearrange
from torch import nn


class HunyuanDiTAttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)

        query = self.q_proj(x[:, :1])
        key = self.k_proj(x)
        value = self.v_proj(x)
        batch_size, _, _ = query.size()

        query = query.reshape(batch_size, -1, self.num_heads, query.size(-1) // self.num_heads).transpose(1, 2)  # (1, H, N, E/H)
        key = key.reshape(batch_size, -1, self.num_heads, key.size(-1) // self.num_heads).transpose(1, 2)  # (L+1, H, N, E/H)
        value = value.reshape(batch_size, -1, self.num_heads, value.size(-1) // self.num_heads).transpose(1, 2)  # (L+1, H, N, E/H)

        x = F.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=None, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, 1, -1)
        x = x.to(query.dtype)
        x = self.c_proj(x) 

        return x.squeeze(1) 


class HunyuanCombinedTimestepTextSizeStyleEmbedding(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim=1024, seq_len=256, cross_attention_dim=2048):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.pooler = HunyuanDiTAttentionPool(
            seq_len, cross_attention_dim, num_heads=8, output_dim=pooled_projection_dim
        )
        # Here we use a default learned embedder layer for future extension.
        self.style_embedder = nn.Embedding(1, embedding_dim)
        extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim
        self.extra_embedder = PixArtAlphaTextProjection(
            in_features=extra_in_dim,
            hidden_size=embedding_dim * 4,
            out_features=embedding_dim,
            act_fn="silu_fp32",
        )

    def forward(self, timestep, encoder_hidden_states, image_meta_size, style, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, 256)

        # extra condition1: text
        pooled_projections = self.pooler(encoder_hidden_states)  # (N, 1024)

        # extra condition2: image meta size embdding
        image_meta_size = get_timestep_embedding(image_meta_size.view(-1), 256, True, 0)
        image_meta_size = image_meta_size.to(dtype=hidden_dtype)
        image_meta_size = image_meta_size.view(-1, 6 * 256)  # (N, 1536)

        # extra condition3: style embedding
        style_embedding = self.style_embedder(style)  # (N, embedding_dim)

        # Concatenate all extra vectors
        extra_cond = torch.cat([pooled_projections, image_meta_size, style_embedding], dim=1)
        conditioning = timesteps_emb + self.extra_embedder(extra_cond)  # [B, D]

        return conditioning


class TimePositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, c, f, h, w = x.size()
        x = rearrange(x, "b c f h w -> (b h w) f c")
        x = x + self.pe[:, :x.size(1)]
        x = rearrange(x, "(b h w) f c -> b c f h w", b=b, h=h, w=w)
        return self.dropout(x)