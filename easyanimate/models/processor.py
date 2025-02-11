from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb
from einops import rearrange, repeat


class HunyuanAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class LazyKVCompressionProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the KVCompression model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        batch_size, channel, num_frames, height, width = hidden_states.shape
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c", f=num_frames, h=height, w=width)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key = rearrange(key, "b (f h w) c -> (b f) c h w", f=num_frames, h=height, w=width)
        key = attn.k_compression(key)
        key_shape = key.size()
        key = rearrange(key, "(b f) c h w -> b (f h w) c", f=num_frames)

        value = rearrange(value, "b (f h w) c -> (b f) c h w", f=num_frames, h=height, w=width)
        value = attn.v_compression(value)
        value = rearrange(value, "(b f) c h w -> b (f h w) c", f=num_frames)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            compression_image_rotary_emb = (
                rearrange(image_rotary_emb[0], "(f h w) c -> f c h w", f=num_frames, h=height, w=width),
                rearrange(image_rotary_emb[1], "(f h w) c -> f c h w", f=num_frames, h=height, w=width),
            )
            compression_image_rotary_emb = (
                F.interpolate(compression_image_rotary_emb[0], size=key_shape[-2:], mode='bilinear'),
                F.interpolate(compression_image_rotary_emb[1], size=key_shape[-2:], mode='bilinear')
            )
            compression_image_rotary_emb = (
                rearrange(compression_image_rotary_emb[0], "f c h w -> (f h w) c"),
                rearrange(compression_image_rotary_emb[1], "f c h w -> (f h w) c"),
            )

            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, compression_image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class EasyAnimateAttnProcessor2_0:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attn2: Attention = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn2 is None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if attn2 is not None:
            query_txt = attn2.to_q(encoder_hidden_states)
            key_txt = attn2.to_k(encoder_hidden_states)
            value_txt = attn2.to_v(encoder_hidden_states)
            
            inner_dim = key_txt.shape[-1]
            head_dim = inner_dim // attn.heads

            query_txt = query_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_txt = key_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_txt = value_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn2.norm_q is not None:
                query_txt = attn2.norm_q(query_txt)
            if attn2.norm_k is not None:
                key_txt = attn2.norm_k(key_txt)

            query = torch.cat([query_txt, query], dim=2)
            key = torch.cat([key_txt, key], dim=2)
            value = torch.cat([value_txt, value], dim=2)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        if attn2 is None:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
        else:
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            encoder_hidden_states = attn2.to_out[0](encoder_hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn2.to_out[1](encoder_hidden_states)
        return hidden_states, encoder_hidden_states

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except:
    print("Flash Attention is not installed. Please install with `pip install flash-attn`, if you want to use SWA.")

class EasyAnimateSWAttnProcessor2_0:
    def __init__(self, cross_attention_size=1024):
        self.cross_attention_size = cross_attention_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        num_frames: int = None, 
        height: int = None, 
        width: int = None,
        attn2: Attention = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        windows_size = height * width

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn2 is None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if attn2 is not None:
            query_txt = attn2.to_q(encoder_hidden_states)
            key_txt = attn2.to_k(encoder_hidden_states)
            value_txt = attn2.to_v(encoder_hidden_states)

            inner_dim = key_txt.shape[-1]
            head_dim = inner_dim // attn.heads

            query_txt = query_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_txt = key_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_txt = value_txt.view(batch_size, -1, attn.heads, head_dim)
            
            if attn2.norm_q is not None:
                query_txt = attn2.norm_q(query_txt)
            if attn2.norm_k is not None:
                key_txt = attn2.norm_k(key_txt)
            
            query = torch.cat([query_txt, query], dim=2)
            key = torch.cat([key_txt, key], dim=2)
            value = torch.cat([value_txt, value], dim=1)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
            
        query = query.transpose(1, 2).to(value)
        key = key.transpose(1, 2).to(value)
        interval = max((query.size(1) - text_seq_length) // (self.cross_attention_size - text_seq_length), 1)

        cross_key = torch.cat([key[:, :text_seq_length], key[:, text_seq_length::interval]], dim=1)
        cross_val = torch.cat([value[:, :text_seq_length], value[:, text_seq_length::interval]], dim=1)
        cross_hidden_states = flash_attn_func(query, cross_key, cross_val, dropout_p=0.0, causal=False)
        
        # Split and rearrange to six directions
        querys = torch.tensor_split(query[:, text_seq_length:], 6, 2)
        keys = torch.tensor_split(key[:, text_seq_length:], 6, 2)
        values = torch.tensor_split(value[:, text_seq_length:], 6, 2)
        
        new_querys = [querys[0]]
        new_keys = [keys[0]]
        new_values = [values[0]]
        for index, mode in enumerate(
            [
                "bs (f h w) hn hd -> bs (f w h) hn hd", 
                "bs (f h w) hn hd -> bs (h f w) hn hd", 
                "bs (f h w) hn hd -> bs (h w f) hn hd", 
                "bs (f h w) hn hd -> bs (w f h) hn hd", 
                "bs (f h w) hn hd -> bs (w h f) hn hd"
            ]
        ):
            new_querys.append(rearrange(querys[index + 1], mode, f=num_frames, h=height, w=width))
            new_keys.append(rearrange(keys[index + 1], mode, f=num_frames, h=height, w=width))
            new_values.append(rearrange(values[index + 1], mode, f=num_frames, h=height, w=width))
        query = torch.cat(new_querys, dim=2)
        key = torch.cat(new_keys, dim=2)
        value = torch.cat(new_values, dim=2)
        
        # apply attention
        hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, causal=False, window_size=(windows_size, windows_size))

        hidden_states = torch.tensor_split(hidden_states, 6, 2)
        new_hidden_states = [hidden_states[0]]
        for index, mode in enumerate(
            [
                "bs (f w h) hn hd -> bs (f h w) hn hd", 
                "bs (h f w) hn hd -> bs (f h w) hn hd", 
                "bs (h w f) hn hd -> bs (f h w) hn hd", 
                "bs (w f h) hn hd -> bs (f h w) hn hd", 
                "bs (w h f) hn hd -> bs (f h w) hn hd"
            ]
        ):
            new_hidden_states.append(rearrange(hidden_states[index + 1], mode, f=num_frames, h=height, w=width))
        hidden_states = torch.cat([cross_hidden_states[:, :text_seq_length], torch.cat(new_hidden_states, dim=2)], dim=1) + cross_hidden_states 

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        if attn2 is None:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
        else:
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            encoder_hidden_states = attn2.to_out[0](encoder_hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn2.to_out[1](encoder_hidden_states)
        return hidden_states, encoder_hidden_states