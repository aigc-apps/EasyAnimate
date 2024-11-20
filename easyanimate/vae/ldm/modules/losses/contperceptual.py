import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?

from ..vaemodules.discriminator import Discriminator3D


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, 
                 outlier_penalty_loss_r=3.0, outlier_penalty_loss_weight=1e5,
                 disc_loss="hinge", l2_loss_weight=0.0, l1_loss_weight=1.0):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator3d = Discriminator3D(
            in_channels=disc_in_channels,
            block_out_channels=(64, 128, 256)
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.outlier_penalty_loss_r = outlier_penalty_loss_r
        self.outlier_penalty_loss_weight = outlier_penalty_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def outlier_penalty_loss(self, posteriors, r):
        batch_size, channels, frames, height, width = posteriors.shape
        mean_X = posteriors.mean(dim=(3, 4), keepdim=True)
        std_X = posteriors.std(dim=(3, 4), keepdim=True)

        diff = torch.abs(posteriors - mean_X)
        penalty = torch.maximum(diff - r * std_X, torch.zeros_like(diff))

        opl = penalty.sum(dim=(3, 4)) / (height * width)
        opl_final = opl.mean(dim=(0, 1, 2))
        return opl_final

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        
        if inputs.ndim==4:
            inputs = inputs.unsqueeze(2)
        if reconstructions.ndim==4:
            reconstructions = reconstructions.unsqueeze(2)

        inputs_ori = inputs
        reconstructions_ori = reconstructions

        # get new loss_weight
        loss_weights = 1
        inputs = inputs.permute(0, 2, 1, 3, 4).flatten(0, 1)
        reconstructions = reconstructions.permute(0, 2, 1, 3, 4).flatten(0, 1)

        rec_loss = 0
        if self.l1_loss_weight > 0:
            rec_loss += torch.abs(inputs.contiguous() - reconstructions.contiguous()) * self.l1_loss_weight
        if self.l2_loss_weight > 0:
            rec_loss += F.mse_loss(inputs.contiguous(), reconstructions.contiguous(), reduction="none") * self.l2_loss_weight
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        rec_loss = rec_loss * loss_weights

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        outlier_penalty_loss = self.outlier_penalty_loss(posteriors.mode(), self.outlier_penalty_loss_r) * self.outlier_penalty_loss_weight

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            logits_fake_3d = self.discriminator3d(reconstructions_ori.contiguous())
            g_loss = -torch.mean(logits_fake) - torch.mean(logits_fake_3d)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    # assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + outlier_penalty_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            logits_real_3d = self.discriminator3d(inputs_ori.contiguous().detach())
            logits_fake_3d = self.discriminator3d(reconstructions_ori.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake) + disc_factor * self.disc_loss(logits_real_3d, logits_fake_3d)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

