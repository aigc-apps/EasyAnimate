from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..util import instantiate_from_config
from .cogvideox_enc_dec import (CogVideoXDecoder3D, CogVideoXEncoder3D,
                                CogVideoXSafeConv3d)


class DiagonalGaussianDistribution:
    def __init__(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic

        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    def sample(self, generator = None) -> torch.FloatTensor:
        x = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * x

    def mode(self):
        return self.mean

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        dims = list(range(1, self.mean.ndim))

        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        dims = list(range(1, self.mean.ndim))

        if self.deterministic:
            return torch.Tensor([0.0])

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

@dataclass
class EncoderOutput:
    latent_dist: DiagonalGaussianDistribution

@dataclass
class DecoderOutput:
    sample: torch.Tensor

def str_eval(item):
    if type(item) == str:
        return eval(item)
    else:
        return item

class AutoencoderKLMagvit_CogVideoX(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        
        mini_batch_encoder=4,
        mini_batch_decoder=1,

        image_key="image",
        train_decoder_only=False,
        train_encoder_only=False,
        monitor=None,
        ckpt_path=None,
        lossconfig=None,
    ):
        super().__init__()
        self.image_key = image_key
        down_block_types = str_eval(down_block_types)
        up_block_types = str_eval(up_block_types)

        self.encoder = CogVideoXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        self.decoder = CogVideoXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.quant_conv = CogVideoXSafeConv3d(2 * out_channels, 2 * out_channels, 1) if use_quant_conv else None
        self.post_quant_conv = CogVideoXSafeConv3d(out_channels, out_channels, 1) if use_post_quant_conv else None

        self.mini_batch_encoder = mini_batch_encoder
        self.mini_batch_decoder = mini_batch_decoder
        self.train_decoder_only = train_decoder_only
        self.train_encoder_only = train_encoder_only
        if train_decoder_only:
            self.encoder.requires_grad_(False)
            if self.quant_conv is not None:
                self.quant_conv.requires_grad_(False)
        if train_encoder_only:
            self.decoder.requires_grad_(False)
            if self.post_quant_conv is not None:
                self.post_quant_conv.requires_grad_(False)
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys="loss")
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)

    def init_from_ckpt(self, path, ignore_keys=list()):
        if path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        m, u = self.load_state_dict(sd, strict=False) # loss.item can be ignored successfully
        print(f"Restored from {path}")
        print(f"missing keys: {str(m)}, unexpected keys: {str(u)}")

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        h = self.encoder(x)
        self.encoder._clear_fake_context_parallel_cache()

        if self.quant_conv is not None:
            moments: torch.Tensor = self.quant_conv(h)
        else:
            moments: torch.Tensor = h
        mean, logvar = moments.chunk(2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)

        return posterior

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        decoded = self.decoder(z)
        self.decoder._clear_fake_context_parallel_cache()
        return decoded

    def forward(self, input, sample_posterior=True):
        if input.ndim==4:
            input = input.unsqueeze(2)
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # print("stt latent shape", z.shape)
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if x.ndim==5:
            x = x.permute(0, 4, 1, 2, 3).to(memory_format=torch.contiguous_format).float()
            return x
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs = self.get_input(batch, self.image_key)
            reconstructions, posterior = self(inputs)
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.train_decoder_only:
            if self.post_quant_conv is not None:
                training_list = list(self.decoder.parameters()) + list(self.post_quant_conv.parameters())
            else:
                training_list = list(self.decoder.parameters())
            opt_ae = torch.optim.AdamW(training_list, lr=lr, betas=(0.9, 0.999), weight_decay=5e-2)
        elif self.train_encoder_only:
            if self.quant_conv is not None:
                training_list = list(self.encoder.parameters()) + list(self.quant_conv.parameters())
            else:
                training_list = list(self.encoder.parameters())
            opt_ae = torch.optim.AdamW(training_list, lr=lr, betas=(0.9, 0.999), weight_decay=5e-2)
        else:
            training_list = list(self.encoder.parameters()) + list(self.decoder.parameters())
            if self.quant_conv is not None:
                training_list = training_list + list(self.quant_conv.parameters())
            if self.post_quant_conv is not None:
                training_list = training_list + list(self.post_quant_conv.parameters())
            opt_ae = torch.optim.AdamW(training_list, lr=lr, betas=(0.9, 0.999), weight_decay=5e-2)
        opt_disc = torch.optim.AdamW(
            list(self.loss.discriminator3d.parameters()) + list(self.loss.discriminator.parameters()),
            lr=lr, betas=(0.9, 0.999), weight_decay=5e-2
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
