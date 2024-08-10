"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/models/motion_module.py
"""
import math

import diffusers
import pkg_resources
import torch

installed_version = diffusers.__version__

if pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version("0.28.2"):
    from diffusers.models.attention_processor import (Attention,
                                                      AttnProcessor2_0,
                                                      HunyuanAttnProcessor2_0)
else:
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from diffusers.models.attention import FeedForward
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn

from .norm import FP32LayerNorm

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_motion_module(
    in_channels,
    motion_module_type: str, 
    motion_module_kwargs: dict,
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs,)
    elif motion_module_type == "VanillaGrid":
        return VanillaTemporalModule(in_channels=in_channels, grid=True, **motion_module_kwargs,)
    else:
        raise ValueError

class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 4096,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
        block_size                         = 1,
        grid                               = False,
        remove_time_embedding_in_photo     = False,

        global_num_attention_heads         = 16,
        global_attention                   = False,
        qk_norm                            = False,
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            grid=grid,
            block_size=block_size,
            remove_time_embedding_in_photo=remove_time_embedding_in_photo,
            qk_norm=qk_norm,
        )
        self.global_transformer = GlobalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=global_num_attention_heads,
            attention_head_dim=in_channels // global_num_attention_heads // temporal_attention_dim_div,
            qk_norm=qk_norm,
        ) if global_attention else None
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)
            if global_attention:
                self.global_transformer.proj_out = zero_module(self.global_transformer.proj_out)

    def forward(self, input_tensor, encoder_hidden_states=None, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)
        if self.global_transformer is not None:
            hidden_states = self.global_transformer(hidden_states)

        output = hidden_states
        return output

class GlobalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        dropout                            = 0.0,
        attention_bias                     = False,
        upcast_attention                   = False,
        qk_norm                            = False,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm1 = FP32LayerNorm(inner_dim)        
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.norm2 = FP32LayerNorm(inner_dim)       
        if pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version("0.28.2"):
            self.attention = Attention(
                query_dim=inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                qk_norm="layer_norm" if qk_norm else None,
                processor=HunyuanAttnProcessor2_0() if qk_norm else AttnProcessor2_0(),
            )
        else:
            self.attention = Attention(
                query_dim=inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.proj_out = nn.Linear(inner_dim, in_channels)
    
    def forward(self, hidden_states):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length, height, width = hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
        
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        # Attention Blocks
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        output = hidden_states + residual
        output = rearrange(output, "b (f h w) c -> b c f h w", f=video_length, h=height, w=width)
        return output

class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 4096,
        grid                               = False,
        block_size                         = 1,
        remove_time_embedding_in_photo     = False,
        qk_norm                            = False,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.block_size = block_size
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    block_size=block_size,
                    grid=grid,
                    remove_time_embedding_in_photo=remove_time_embedding_in_photo,
                    qk_norm=qk_norm
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)    
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, height=height, weight=weight)

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        return output

class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 4096,
        block_size                         = 1,
        grid                               = False,
        remove_time_embedding_in_photo     = False,
        qk_norm                            = False,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    block_size=block_size,
                    grid=grid,
                    remove_time_embedding_in_photo=remove_time_embedding_in_photo,
                    qk_norm="layer_norm" if qk_norm else None,
                    processor=HunyuanAttnProcessor2_0() if qk_norm else AttnProcessor2_0(),
                ) if pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version("0.28.2") else \
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    block_size=block_size,
                    grid=grid,
                    remove_time_embedding_in_photo=remove_time_embedding_in_photo,
                )
            )
            norms.append(FP32LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = FP32LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, height=None, weight=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states, 
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None, 
                video_length=video_length, 
                height=height, 
                weight=weight, 
            ) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 4096
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class VersatileAttention(Attention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 4096,  
            grid                               = False,
            block_size                         = 1,
            remove_time_embedding_in_photo     = False,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal" or attention_mode == "Global"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
        self.block_size = block_size
        self.grid = grid
        self.remove_time_embedding_in_photo = remove_time_embedding_in_photo
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") or (temporal_position_encoding and attention_mode == "Global") else None

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, height=None, weight=None):
        batch_size, sequence_length, _ = hidden_states.shape
    
        if self.attention_mode == "Temporal":
            # for add pos_encoder 
            _, before_d, _c = hidden_states.size()
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            if self.remove_time_embedding_in_photo:
                if self.pos_encoder is not None and video_length > 1:
                    hidden_states = self.pos_encoder(hidden_states)
            else:
                if self.pos_encoder is not None:
                    hidden_states = self.pos_encoder(hidden_states)
            
            if self.grid:
                hidden_states = rearrange(hidden_states, "(b d) f c -> b f d c", f=video_length, d=before_d)
                hidden_states = rearrange(hidden_states, "b f (h w) c -> b f h w c", h=height, w=weight)

                hidden_states = rearrange(hidden_states, "b f (h n) (w m) c -> (b h w) (f n m) c", n=self.block_size, m=self.block_size)
                d = before_d // self.block_size // self.block_size
            else:
                d = before_d    
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        elif self.attention_mode == "Global":
            # for add pos_encoder 
            _, d, _c = hidden_states.size()
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)
            hidden_states = rearrange(hidden_states, "(b d) f c -> b (f d) c", f=video_length, d=d)
        else:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        bs = 512
        new_hidden_states = []
        for i in range(0, hidden_states.shape[0], bs):
            __hidden_states = super().forward(
                hidden_states[i : i + bs],
                encoder_hidden_states=encoder_hidden_states[i : i + bs],
                attention_mask=attention_mask
            )
            new_hidden_states.append(__hidden_states)
        hidden_states = torch.cat(new_hidden_states, dim = 0)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            if self.grid:
                hidden_states = rearrange(hidden_states, "(b f n m) (h w) c -> (b f) h n w m c", f=video_length, n=self.block_size, m=self.block_size, h=height // self.block_size, w=weight // self.block_size)
                hidden_states = rearrange(hidden_states, "b h n w m c -> b (h n) (w m) c")
                hidden_states = rearrange(hidden_states, "b h w c -> b (h w) c")
        elif self.attention_mode == "Global":
            hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=video_length, d=d)

        return hidden_states