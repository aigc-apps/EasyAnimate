import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple
from einops import rearrange, repeat
from diffusers.utils import (USE_PEFT_BACKEND, BaseOutput, is_torch_version,
                             logging)
from ..modules.vaemodules.activations import get_activation
from ..modules.vaemodules.common import CausalConv3d
from ..modules.vaemodules.down_blocks import get_down_block
from ..modules.vaemodules.mid_blocks import get_mid_block
from ..modules.vaemodules.up_blocks import get_up_block


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 8):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialDownBlock3D",)`):
            The types of down blocks to use. 
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        use_gc_blocks (`Tuple[bool, ...]`, *optional*, defaults to `None`):
            Whether to use global context blocks for each down block.
        mid_block_type (`str`, *optional*, defaults to `"MidBlock3D"`):
            The type of mid block to use. 
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        num_attention_heads (`int`, *optional*, defaults to 1):
            The number of attention heads to use.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 8,
        down_block_types = ("SpatialDownBlock3D",),
        ch = 128,
        ch_mult = [1,2,4,4,],
        use_gc_blocks = None,
        mid_block_type: str = "MidBlock3D",
        mid_block_use_attention: bool = True,
        mid_block_attention_type: str = "3d",
        mid_block_num_attention_heads: int = 1,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        num_attention_heads: int = 1,
        double_z: bool = True,
        slice_mag_vae: bool = False,
        slice_compression_vae: bool = False,
        cache_compression_vae: bool = False,
        spatial_group_norm: bool = False,
        mini_batch_encoder: int = 9,
        verbose = False,
    ):
        super().__init__()
        block_out_channels = [ch * i for i in ch_mult]
        assert len(down_block_types) == len(block_out_channels), (
            "Number of down block types must match number of block output channels."
        )
        if use_gc_blocks is not None:
            assert len(use_gc_blocks) == len(down_block_types), (
                "Number of GC blocks must match number of down block types."
            )
        else:
            use_gc_blocks = [False] * len(down_block_types)
        self.conv_in = CausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
        )

        self.down_blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channels = output_channels
            output_channels = block_out_channels[i]
            is_final_block = (i == len(block_out_channels) - 1)
            down_block = get_down_block(
                down_block_type,
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=1e-6,
                num_attention_heads=num_attention_heads,
                add_gc_block=use_gc_blocks[i],
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)

        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
            add_attention=mid_block_use_attention,
            attention_type=mid_block_attention_type,
            num_attention_heads=mid_block_num_attention_heads,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

        self.slice_mag_vae = slice_mag_vae
        self.slice_compression_vae = slice_compression_vae
        self.cache_compression_vae = cache_compression_vae
        self.mini_batch_encoder = mini_batch_encoder
        self.spatial_group_norm = spatial_group_norm
        self.verbose = verbose

    def set_padding_one_frame(self):
        def _set_padding_one_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 1
            for sub_name, sub_mod in module.named_children():
                _set_padding_one_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_padding_one_frame(name, module)

    def set_padding_more_frame(self):
        def _set_padding_more_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 2
            for sub_name, sub_mod in module.named_children():
                _set_padding_more_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_padding_more_frame(name, module)

    def set_cache_slice_vae_padding_one_frame(self):
        def _set_cache_slice_vae_padding_one_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 5
            for sub_name, sub_mod in module.named_children():
                _set_cache_slice_vae_padding_one_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_cache_slice_vae_padding_one_frame(name, module)
            
    def set_cache_slice_vae_padding_more_frame(self):
        def _set_cache_slice_vae_padding_more_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 6
            for sub_name, sub_mod in module.named_children():
                _set_cache_slice_vae_padding_more_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_cache_slice_vae_padding_more_frame(name, module)

    def set_3dgroupnorm_for_submodule(self):
        def _set_3dgroupnorm_for_submodule(name, module):
            if hasattr(module, 'set_3dgroupnorm'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.set_3dgroupnorm = True
            for sub_name, sub_mod in module.named_children():
                _set_3dgroupnorm_for_submodule(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_3dgroupnorm_for_submodule(name, module)

    def single_forward(self, x: torch.Tensor, previous_features: torch.Tensor, after_features: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if previous_features is not None and after_features is None:
            x = torch.concat([previous_features, x], 2)
        elif previous_features is None and after_features is not None:
            x = torch.concat([x, after_features], 2)
        elif previous_features is not None and after_features is not None:
            x = torch.concat([previous_features, x, after_features], 2)

        x = self.conv_in(x)

        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)

        if self.spatial_group_norm:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv_norm_out(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)
        else:
            x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        if previous_features is not None and after_features is None:
            x = x[:, :, 1:]
        elif previous_features is None and after_features is not None:
            x = x[:, :, :2]
        elif previous_features is not None and after_features is not None:
            x = x[:, :, 1:3]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_group_norm:
            self.set_3dgroupnorm_for_submodule()

        if self.cache_compression_vae:
            _, _, f, _, _ = x.size()
            if f % 2 != 0:
                self.set_padding_one_frame()
                first_frames = self.single_forward(x[:, :, 0:1, :, :], None, None)
                self.set_padding_more_frame()

                new_pixel_values = [first_frames]
                start_index = 1
            else:
                self.set_padding_more_frame()
                new_pixel_values = []
                start_index = 0

            for i in range(start_index, x.shape[2], self.mini_batch_encoder):
                next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_encoder, :, :], None, None)
                new_pixel_values.append(next_frames)
            new_pixel_values = torch.cat(new_pixel_values, dim=2)
        elif self.slice_compression_vae:
            _, _, f, _, _ = x.size()
            if f % 2 != 0:
                self.set_padding_one_frame()
                first_frames = self.single_forward(x[:, :, 0:1, :, :], None, None)
                self.set_padding_more_frame()

                new_pixel_values = [first_frames]
                start_index = 1
            else:
                self.set_padding_more_frame()
                new_pixel_values = []
                start_index = 0

            for i in range(start_index, x.shape[2], self.mini_batch_encoder):
                next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_encoder, :, :], None, None)
                new_pixel_values.append(next_frames)
            new_pixel_values = torch.cat(new_pixel_values, dim=2)
        elif self.slice_mag_vae:
            _, _, f, _, _ = x.size()
            new_pixel_values = []
            for i in range(0, x.shape[2], self.mini_batch_encoder):
                next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_encoder, :, :], None, None)
                new_pixel_values.append(next_frames)
            new_pixel_values = torch.cat(new_pixel_values, dim=2)
        else:
            new_pixel_values = self.single_forward(x, None, None)
        return new_pixel_values

class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 8):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("SpatialUpBlock3D",)`):
            The types of up blocks to use. 
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        use_gc_blocks (`Tuple[bool, ...]`, *optional*, defaults to `None`):
            Whether to use global context blocks for each down block.
        mid_block_type (`str`, *optional*, defaults to `"MidBlock3D"`):
            The type of mid block to use. 
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        num_attention_heads (`int`, *optional*, defaults to 1):
            The number of attention heads to use.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        up_block_types  = ("SpatialUpBlock3D",),
        ch = 128,
        ch_mult = [1,2,4,4,],
        use_gc_blocks = None,
        mid_block_type: str = "MidBlock3D",
        mid_block_use_attention: bool = True,
        mid_block_attention_type: str = "3d",
        mid_block_num_attention_heads: int = 1,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        num_attention_heads: int = 1,
        slice_mag_vae: bool = False,
        slice_compression_vae: bool = False,
        cache_compression_vae: bool = False,
        spatial_group_norm: bool = False,
        mini_batch_decoder: int = 3, 
        verbose = False,
    ):
        super().__init__()
        block_out_channels = [ch * i for i in ch_mult]
        assert len(up_block_types) == len(block_out_channels), (
            "Number of up block types must match number of block output channels."
        )
        if use_gc_blocks is not None:
            assert len(use_gc_blocks) == len(up_block_types), (
                "Number of GC blocks must match number of up block types."
            )
        else:
            use_gc_blocks = [False] * len(up_block_types)

        self.conv_in = CausalConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
        )

        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-6,
            add_attention=mid_block_use_attention,
            attention_type=mid_block_attention_type,
            num_attention_heads=mid_block_num_attention_heads,
        )

        self.up_blocks = nn.ModuleList([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channels = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            input_channels = output_channels
            output_channels = reversed_block_out_channels[i]
            # is_first_block = i == 0
            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=layers_per_block + 1,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                norm_eps=1e-6,
                num_attention_heads=num_attention_heads,
                add_gc_block=use_gc_blocks[i],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(act_fn)

        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3)
        
        self.slice_mag_vae = slice_mag_vae
        self.slice_compression_vae = slice_compression_vae
        self.cache_compression_vae = cache_compression_vae
        self.mini_batch_decoder = mini_batch_decoder
        self.spatial_group_norm = spatial_group_norm
        self.verbose = verbose

    def set_padding_one_frame(self):
        def _set_padding_one_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 1
            for sub_name, sub_mod in module.named_children():
                _set_padding_one_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_padding_one_frame(name, module)

    def set_padding_more_frame(self):
        def _set_padding_more_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 2
            for sub_name, sub_mod in module.named_children():
                _set_padding_more_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_padding_more_frame(name, module)

    def set_cache_slice_vae_padding_one_frame(self):
        def _set_cache_slice_vae_padding_one_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 5
            for sub_name, sub_mod in module.named_children():
                _set_cache_slice_vae_padding_one_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_cache_slice_vae_padding_one_frame(name, module)
            
    def set_cache_slice_vae_padding_more_frame(self):
        def _set_cache_slice_vae_padding_more_frame(name, module):
            if hasattr(module, 'padding_flag'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.padding_flag = 6
            for sub_name, sub_mod in module.named_children():
                _set_cache_slice_vae_padding_more_frame(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_cache_slice_vae_padding_more_frame(name, module)

    def set_3dgroupnorm_for_submodule(self):
        def _set_3dgroupnorm_for_submodule(name, module):
            if hasattr(module, 'set_3dgroupnorm'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.set_3dgroupnorm = True
            for sub_name, sub_mod in module.named_children():
                _set_3dgroupnorm_for_submodule(sub_name, sub_mod)
        for name, module in self.named_children():
            _set_3dgroupnorm_for_submodule(name, module)
            
    def clear_cache(self):
        def _clear_cache(name, module):
            if hasattr(module, 'prev_features'):
                if self.verbose:
                    print('Set pad mode for module[%s] type=%s' % (name, str(type(module))))
                module.prev_features = None
            for sub_name, sub_mod in module.named_children():
                _clear_cache(sub_name, sub_mod)
        for name, module in self.named_children():
            _clear_cache(name, module)
            
    def single_forward(self, x: torch.Tensor, previous_features: torch.Tensor, after_features: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        if previous_features is not None and after_features is None:
            b, c, t, h, w = x.size()
            x = torch.concat([previous_features, x], 2)
            x = self.conv_in(x)
            x = self.mid_block(x)
            x = x[:, :, -t:]
        elif previous_features is None and after_features is not None:
            b, c, t, h, w = x.size()
            x = torch.concat([x, after_features], 2)
            x = self.conv_in(x)
            x = self.mid_block(x)
            x = x[:, :, :t]
        elif previous_features is not None and after_features is not None:
            _, _, t_1, _, _ = previous_features.size()
            _, _, t_2, _, _ = x.size()
            x = torch.concat([previous_features, x, after_features], 2)
            x = self.conv_in(x)
            x = self.mid_block(x)
            x = x[:, :, t_1:(t_1 + t_2)]
        else:
            x = self.conv_in(x)
            x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        if self.spatial_group_norm:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv_norm_out(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)
        else:
            x = self.conv_norm_out(x)

        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_group_norm:
            self.set_3dgroupnorm_for_submodule()

        if self.cache_compression_vae:
            _, _, f, _, _ = x.size()
            if f == 1:
                self.set_padding_one_frame()
                first_frames = self.single_forward(x[:, :, :1, :, :], None, None)
                new_pixel_values = [first_frames]
                start_index = 1
            else:
                self.set_cache_slice_vae_padding_one_frame()
                first_frames = self.single_forward(x[:, :, :self.mini_batch_decoder, :, :], None, None)
                new_pixel_values = [first_frames]
                start_index = self.mini_batch_decoder

            for i in range(start_index, x.shape[2], self.mini_batch_decoder):
                self.set_cache_slice_vae_padding_more_frame()
                next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_decoder, :, :], None, None)
                new_pixel_values.append(next_frames)
            new_pixel_values = torch.cat(new_pixel_values, dim=2)
        elif self.slice_compression_vae:
            _, _, f, _, _ = x.size()
            if f % 2 != 0:
                self.set_padding_one_frame()
                first_frames = self.single_forward(x[:, :, 0:1, :, :], None, None)
                self.set_padding_more_frame()
                new_pixel_values = [first_frames]
                start_index = 1
            else:
                self.set_padding_more_frame()
                new_pixel_values = []
                start_index = 0
                
            previous_features = None
            for i in range(start_index, x.shape[2], self.mini_batch_decoder):
                after_features = x[:, :, i + self.mini_batch_decoder: i + 2 * self.mini_batch_decoder, :, :] if i + self.mini_batch_decoder < x.shape[2] else None
                next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_decoder, :, :], previous_features, after_features)
                previous_features = x[:, :, i: i + self.mini_batch_decoder, :, :]
                new_pixel_values.append(next_frames)
            new_pixel_values = torch.cat(new_pixel_values, dim=2)
        elif self.slice_mag_vae:
            _, _, f, _, _ = x.size()
            new_pixel_values = []
            for i in range(0, x.shape[2], self.mini_batch_decoder):
                next_frames = self.single_forward(x[:, :, i: i + self.mini_batch_decoder, :, :], None, None)
                new_pixel_values.append(next_frames)
            new_pixel_values = torch.cat(new_pixel_values, dim=2)
        else:
            new_pixel_values = self.single_forward(x, None, None)
        return new_pixel_values
