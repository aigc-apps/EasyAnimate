# Copyright 2024 EasyAnimate Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, HunyuanDiT2DModel
from diffusers.models.embeddings import (get_2d_rotary_pos_embed,
                                         get_3d_rotary_pos_embed)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (is_torch_xla_available, logging,
                             replace_example_docstring)
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from transformers import (BertModel, BertTokenizer, CLIPImageProcessor,
                          CLIPVisionModelWithProjection, T5Tokenizer,
                          T5EncoderModel)

from .pipeline_easyanimate import EasyAnimatePipelineOutput
from ..models import AutoencoderKLMagvit, EasyAnimateTransformer3DModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> pass
        ```
"""


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def add_noise_to_reference_video(image, ratio=None):
    if ratio is None:
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
        sigma = torch.exp(sigma).to(image.dtype)
    else:
        sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
    
    image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image


class EasyAnimatePipeline_Multi_Text_Encoder_Inpaint(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using EasyAnimate.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    EasyAnimate uses two text encoders: [mT5](https://huggingface.co/google/mt5-base) and [bilingual CLIP](fine-tuned by
    HunyuanDiT team)

    Args:
        vae ([`AutoencoderKLMagvit`]):
            Variational Auto-Encoder (VAE) Model to encode and decode video to and from latent representations. 
        text_encoder (Optional[`~transformers.BertModel`, `~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
            EasyAnimate uses a fine-tuned [bilingual CLIP].
        tokenizer (Optional[`~transformers.BertTokenizer`, `~transformers.CLIPTokenizer`]):
            A `BertTokenizer` or `CLIPTokenizer` to tokenize text.
        transformer ([`EasyAnimateTransformer3DModel`]):
            The EasyAnimate model designed by Tencent Hunyuan.
        text_encoder_2 (`T5EncoderModel`):
            The mT5 embedder. 
        tokenizer_2 (`T5Tokenizer`):
            The tokenizer for the mT5 embedder.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with EasyAnimate to denoise the encoded image latents.
        clip_image_processor (`CLIPImageProcessor`):
            The CLIP image embedder. 
        clip_image_encoder (`CLIPVisionModelWithProjection`):
            The image processor for the CLIP image embedder.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = [
        "safety_checker",
        "feature_extractor",
        "text_encoder_2",
        "tokenizer_2",
        "text_encoder",
        "tokenizer",
        "clip_image_encoder",
    ]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_2",
        "negative_prompt_embeds_2",
    ]

    def __init__(
        self,
        vae: AutoencoderKLMagvit,
        text_encoder: BertModel,
        tokenizer: BertTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5Tokenizer,
        transformer: EasyAnimateTransformer3DModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        clip_image_processor: CLIPImageProcessor = None,
        clip_image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            text_encoder_2=text_encoder_2,
            clip_image_processor=clip_image_processor, 
            clip_image_encoder=clip_image_encoder,
        )

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_sequential_cpu_offload(self, *args, **kwargs):
        super().enable_sequential_cpu_offload(*args, **kwargs)
        if hasattr(self.transformer, "clip_projection") and self.transformer.clip_projection is not None:
            import accelerate
            accelerate.hooks.remove_hook_from_module(self.transformer.clip_projection, recurse=True)
            self.transformer.clip_projection = self.transformer.clip_projection.to("cuda")

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        text_encoder_index: int = 0,
        actual_max_sequence_length: int = 256
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            dtype (`torch.dtype`):
                torch dtype
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
            text_encoder_index (`int`, *optional*):
                Index of the text encoder to use. `0` for clip and `1` for T5.
        """
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = tokenizers[text_encoder_index]
        text_encoder = text_encoders[text_encoder_index]

        if max_sequence_length is None:
            if text_encoder_index == 0:
                max_length = min(self.tokenizer.model_max_length, actual_max_sequence_length)
            if text_encoder_index == 1:
                max_length = min(self.tokenizer_2.model_max_length, actual_max_sequence_length)
        else:
            max_length = max_sequence_length

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if text_input_ids.shape[-1] > actual_max_sequence_length:
                reprompt = tokenizer.batch_decode(text_input_ids[:, :actual_max_sequence_length], skip_special_tokens=True)
                text_inputs = tokenizer(
                    reprompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                _actual_max_sequence_length = min(tokenizer.model_max_length, actual_max_sequence_length)
                removed_text = tokenizer.batch_decode(untruncated_ids[:, _actual_max_sequence_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {_actual_max_sequence_length} tokens: {removed_text}"
                )
            prompt_attention_mask = text_inputs.attention_mask.to(device)
            if self.transformer.config.enable_text_attention_mask:
                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    attention_mask=prompt_attention_mask,
                )
            else:
                prompt_embeds = text_encoder(
                    text_input_ids.to(device)
                )
            prompt_embeds = prompt_embeds[0]
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids
            if uncond_input_ids.shape[-1] > actual_max_sequence_length:
                reuncond_tokens = tokenizer.batch_decode(uncond_input_ids[:, :actual_max_sequence_length], skip_special_tokens=True)
                uncond_input = tokenizer(
                    reuncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                uncond_input_ids = uncond_input.input_ids

            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            if self.transformer.config.enable_text_attention_mask:
                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=negative_prompt_attention_mask,
                )
            else:
                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device)
                )
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        prompt_embeds_2=None,
        negative_prompt_embeds_2=None,
        prompt_attention_mask_2=None,
        negative_prompt_attention_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is None and prompt_embeds_2 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_2`. Cannot leave both `prompt` and `prompt_embeds_2` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if prompt_embeds_2 is not None and prompt_attention_mask_2 is None:
            raise ValueError("Must provide `prompt_attention_mask_2` when specifying `prompt_embeds_2`.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if negative_prompt_embeds_2 is not None and negative_prompt_attention_mask_2 is None:
            raise ValueError(
                "Must provide `negative_prompt_attention_mask_2` when specifying `negative_prompt_embeds_2`."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        if prompt_embeds_2 is not None and negative_prompt_embeds_2 is not None:
            if prompt_embeds_2.shape != negative_prompt_embeds_2.shape:
                raise ValueError(
                    "`prompt_embeds_2` and `negative_prompt_embeds_2` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_2` {prompt_embeds_2.shape} != `negative_prompt_embeds_2`"
                    f" {negative_prompt_embeds_2.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        if mask is not None:
            mask = mask.to(device=device, dtype=dtype)
            if self.vae.quant_conv is None or self.vae.quant_conv.weight.ndim==5:
                bs = 1
                new_mask = []
                for i in range(0, mask.shape[0], bs):
                    mask_bs = mask[i : i + bs]
                    mask_bs = self.vae.encode(mask_bs)[0]
                    mask_bs = mask_bs.mode()
                    new_mask.append(mask_bs)
                mask = torch.cat(new_mask, dim = 0)
                mask = mask * self.vae.config.scaling_factor

            else:
                if mask.shape[1] == 4:
                    mask = mask
                else:
                    video_length = mask.shape[2]
                    mask = rearrange(mask, "b c f h w -> (b f) c h w")
                    mask = self._encode_vae_image(mask, generator=generator)
                    mask = rearrange(mask, "(b f) c h w -> b c f h w", f=video_length)

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=dtype)
            if self.transformer.config.add_noise_in_inpaint_model:
                masked_image = add_noise_to_reference_video(masked_image, ratio=noise_aug_strength)
            if self.vae.quant_conv is None or self.vae.quant_conv.weight.ndim==5:
                bs = 1
                new_mask_pixel_values = []
                for i in range(0, masked_image.shape[0], bs):
                    mask_pixel_values_bs = masked_image[i : i + bs]
                    mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                    mask_pixel_values_bs = mask_pixel_values_bs.mode()
                    new_mask_pixel_values.append(mask_pixel_values_bs)
                masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
                masked_image_latents = masked_image_latents * self.vae.config.scaling_factor

            else:
                if masked_image.shape[1] == 4:
                    masked_image_latents = masked_image
                else:
                    video_length = masked_image.shape[2]
                    masked_image = rearrange(masked_image, "b c f h w -> (b f) c h w")
                    masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
                    masked_image_latents = rearrange(masked_image_latents, "(b f) c h w -> b c f h w", f=video_length)

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def prepare_latents(
        self, 
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        video=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_video_latents=False,
    ):
        if self.vae.quant_conv is None or self.vae.quant_conv.weight.ndim==5:
            if self.vae.cache_mag_vae:
                mini_batch_encoder = self.vae.mini_batch_encoder
                mini_batch_decoder = self.vae.mini_batch_decoder
                shape = (batch_size, num_channels_latents, int((video_length - 1) // mini_batch_encoder * mini_batch_decoder + 1) if video_length != 1 else 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
            else:
                mini_batch_encoder = self.vae.mini_batch_encoder
                mini_batch_decoder = self.vae.mini_batch_decoder
                shape = (batch_size, num_channels_latents, int(video_length // mini_batch_encoder * mini_batch_decoder) if video_length != 1 else 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
        else:
            shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if return_video_latents or (latents is None and not is_strength_max):
            video = video.to(device=device, dtype=dtype)
            if self.vae.quant_conv is None or self.vae.quant_conv.weight.ndim==5:
                bs = 1
                new_video = []
                for i in range(0, video.shape[0], bs):
                    video_bs = video[i : i + bs]
                    video_bs = self.vae.encode(video_bs)[0]
                    video_bs = video_bs.sample()
                    new_video.append(video_bs)
                video = torch.cat(new_video, dim = 0)
                video = video * self.vae.config.scaling_factor

            else:
                if video.shape[1] == 4:
                    video = video
                else:
                    video_length = video.shape[2]
                    video = rearrange(video, "b c f h w -> (b f) c h w")
                    video = self._encode_vae_image(video, generator=generator)
                    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
            video_latents = video.repeat(batch_size // video.shape[0], 1, 1, 1, 1)
            video_latents = video_latents.to(device=device, dtype=dtype)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(video_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        # scale the initial noise by the standard deviation required by the scheduler
        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_video_latents:
            outputs += (video_latents,)

        return outputs

    def smooth_output(self, video, mini_batch_encoder, mini_batch_decoder):
        if video.size()[2] <= mini_batch_encoder:
            return video
        prefix_index_before = mini_batch_encoder // 2
        prefix_index_after = mini_batch_encoder - prefix_index_before
        pixel_values = video[:, :, prefix_index_before:-prefix_index_after]

        # Encode middle videos
        latents = self.vae.encode(pixel_values)[0]
        latents = latents.mode()
        # Decode middle videos
        middle_video = self.vae.decode(latents)[0]

        video[:, :, prefix_index_before:-prefix_index_after] = (video[:, :, prefix_index_before:-prefix_index_after] + middle_video) / 2
        return video

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        if self.vae.quant_conv is None or self.vae.quant_conv.weight.ndim==5:
            mini_batch_encoder = self.vae.mini_batch_encoder
            mini_batch_decoder = self.vae.mini_batch_decoder
            video = self.vae.decode(latents)[0]
            video = video.clamp(-1, 1)
            if not self.vae.cache_compression_vae and not self.vae.cache_mag_vae:
                video = self.smooth_output(video, mini_batch_encoder, mini_batch_decoder).cpu().clamp(-1, 1)
        else:
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            video = []
            for frame_idx in tqdm(range(latents.shape[0])):
                video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        video_length: Optional[int] = None,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        masked_video_latents: Union[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "latent",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = (1024, 1024),
        target_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        clip_image: Image = None,
        clip_apply_ratio: float = 0.40,
        strength: float = 1.0,
        noise_aug_strength: float = 0.0563,
        comfyui_progressbar: bool = False,
    ):
        r"""
        The call function to the pipeline for generation with HunyuanDiT.

        Examples:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            video_length (`int`, *optional*):
                Length of the video to be generated in seconds. This parameter influences the number of frames and
                continuity of generated content.
            video (`torch.FloatTensor`, *optional*):
                A tensor representing an input video, which can be modified depending on the prompts provided.
            mask_video (`torch.FloatTensor`, *optional*):
                A tensor to specify areas of the video to be masked (omitted from generation).
            masked_video_latents (`torch.FloatTensor`, *optional*):
                Latents from masked portions of the video, utilized during image generation.
            height (`int`, *optional*):
                The height in pixels of the generated image or video frames.
            width (`int`, *optional*):
                The width in pixels of the generated image or video frames.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image but slower
                inference time. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text 
                `prompt` at the expense of lower image quality. Guidance scale is effective when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to exclude in image generation. If not defined, you need to
                provide `negative_prompt_embeds`. This parameter is ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                A parameter defined in the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies to the
                [`~schedulers.DDIMScheduler`] and is ignored in other schedulers. It adjusts noise level during the 
                inference process.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) for setting
                random seeds which helps in making generation deterministic.
            latents (`torch.Tensor`, *optional*):
                A pre-computed latent representation which can be used to guide the generation process.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, embeddings are generated from the `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Secondary set of pre-generated text embeddings, useful for advanced prompt weighting.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings, aiding in fine-tuning what should not be represented in the outputs.
                If not provided, embeddings are generated from the `negative_prompt` argument.
            negative_prompt_embeds_2 (`torch.Tensor`, *optional*):
                Secondary set of pre-generated negative text embeddings for further control.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask guiding the focus of the model on specific parts of the prompt text. Required when using
                `prompt_embeds`.
            prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the secondary prompt embedding.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt, needed when `negative_prompt_embeds` are used.
            negative_prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the secondary negative prompt embedding.
            output_type (`str`, *optional*, defaults to `"latent"`):
                The output format of the generated image. Choose between `PIL.Image` and `np.array` to define
                how you want the results to be formatted.
            return_dict (`bool`, *optional*, defaults to `True`):
                If set to `True`, a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] will be returned;
                otherwise, a tuple containing the generated images and safety flags will be returned.
            callback_on_step_end (`Callable[[int, int, Dict], None]`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A callback function (or a list of them) that will be executed at the end of each denoising step,
                allowing for custom processing during generation.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Specifies which tensor inputs should be included in the callback function. If not defined, all tensor
                inputs will be passed, facilitating enhanced logging or monitoring of the generation process.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Rescale parameter for adjusting noise configuration based on guidance rescale. Based on findings from
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            original_size (`Tuple[int, int]`, *optional*, defaults to `(1024, 1024)`):
                The original dimensions of the image. Used to compute time ids during the generation process.
            target_size (`Tuple[int, int]`, *optional*):
                The targeted dimensions of the generated image, also utilized in the time id calculations.
            crops_coords_top_left (`Tuple[int, int]`, *optional*, defaults to `(0, 0)`):
                Coordinates defining the top left corner of any cropping, utilized while calculating the time ids.
            clip_image (`Image`, *optional*):
                An optional image to assist in the generation process. It may be used as an additional visual cue.
            clip_apply_ratio (`float`, *optional*, defaults to 0.40):
                Ratio indicating how much influence the clip image should exert over the generated content.
            strength (`float`, *optional*, defaults to 1.0):
                Affects the overall styling or quality of the generated output. Values closer to 1 usually provide direct 
                adherence to prompts.
            comfyui_progressbar (`bool`, *optional*, defaults to `False`):
                Enables a progress bar in ComfyUI, providing visual feedback during the generation process.

        Examples:
            # Example usage of the function for generating images based on prompts.
        
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                Returns either a structured output containing generated images and their metadata when `return_dict` is
                `True`, or a simpler tuple, where the first element is a list of generated images and the second
                element indicates if any of them contain "not-safe-for-work" (NSFW) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        height = int(height // 16 * 16)
        width = int(width // 16 * 16)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.text_encoder_2 is not None:
            dtype = self.text_encoder_2.dtype
        else:
            dtype = self.transformer.dtype
            
        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_index=0,
        )
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds_2,
            negative_prompt_embeds=negative_prompt_embeds_2,
            prompt_attention_mask=prompt_attention_mask_2,
            negative_prompt_attention_mask=negative_prompt_attention_mask_2,
            text_encoder_index=1,
        )

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 3)
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        if video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        # Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_transformer = self.transformer.config.in_channels
        return_image_latents = num_channels_transformer == num_channels_latents

        # 5. Prepare latents.
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            dtype,
            device,
            generator,
            latents,
            video=init_video,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_video_latents=return_image_latents,
        )
        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare clip latents if it needs.
        if clip_image is not None and self.transformer.enable_clip_in_inpaint:
            inputs = self.clip_image_processor(images=clip_image, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=dtype)
            clip_encoder_hidden_states = self.clip_image_encoder(**inputs).last_hidden_state[:, 1:]
            clip_encoder_hidden_states_neg = torch.zeros(
                [
                    batch_size, 
                    int(self.clip_image_encoder.config.image_size / self.clip_image_encoder.config.patch_size) ** 2, 
                    int(self.clip_image_encoder.config.hidden_size)
                ]
            ).to(device, dtype=dtype)

            clip_attention_mask = torch.ones([batch_size, self.transformer.n_query]).to(device, dtype=dtype)
            clip_attention_mask_neg = torch.zeros([batch_size, self.transformer.n_query]).to(device, dtype=dtype)

            clip_encoder_hidden_states_input = torch.cat([clip_encoder_hidden_states_neg, clip_encoder_hidden_states]) if self.do_classifier_free_guidance else clip_encoder_hidden_states
            clip_attention_mask_input = torch.cat([clip_attention_mask_neg, clip_attention_mask]) if self.do_classifier_free_guidance else clip_attention_mask

        elif clip_image is None and num_channels_transformer != num_channels_latents and self.transformer.enable_clip_in_inpaint:
            clip_encoder_hidden_states = torch.zeros(
                [
                    batch_size, 
                    int(self.clip_image_encoder.config.image_size / self.clip_image_encoder.config.patch_size) ** 2, 
                    int(self.clip_image_encoder.config.hidden_size)
                ]
            ).to(device, dtype=dtype)

            clip_attention_mask = torch.zeros([batch_size, self.transformer.n_query])
            clip_attention_mask = clip_attention_mask.to(device, dtype=dtype)

            clip_encoder_hidden_states_input = torch.cat([clip_encoder_hidden_states] * 2) if self.do_classifier_free_guidance else clip_encoder_hidden_states
            clip_attention_mask_input = torch.cat([clip_attention_mask] * 2) if self.do_classifier_free_guidance else clip_attention_mask

        else:
            clip_encoder_hidden_states_input = None
            clip_attention_mask_input = None
        if comfyui_progressbar:
            pbar.update(1)

        # 7. Prepare inpaint latents if it needs.
        if mask_video is not None:
            if (mask_video == 255).all():
                # Use zero latents if we want to t2v.
                if self.transformer.resize_inpaint_mask_directly:
                    mask_latents = torch.zeros_like(latents)[:, :1].to(device, dtype)
                else:
                    mask_latents = torch.zeros_like(latents).to(device, dtype)
                masked_video_latents = torch.zeros_like(latents).to(device, dtype)

                mask_input = torch.cat([mask_latents] * 2) if self.do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if self.do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(dtype)
            else:
                # Prepare mask latent variables
                video_length = video.shape[2]
                mask_condition = self.mask_processor.preprocess(rearrange(mask_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    if masked_video_latents is None:
                        masked_video = init_video * (mask_condition_tile < 0.5) + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                    else:
                        masked_video = masked_video_latents
                    
                    if self.transformer.resize_inpaint_mask_directly:
                        _, masked_video_latents = self.prepare_mask_latents(
                            None,
                            masked_video,
                            batch_size,
                            height,
                            width,
                            dtype,
                            device,
                            generator,
                            self.do_classifier_free_guidance,
                            noise_aug_strength=noise_aug_strength,
                        )
                        mask_latents = resize_mask(1 - mask_condition, masked_video_latents, self.vae.cache_mag_vae)
                        mask_latents = mask_latents.to(device, dtype) * self.vae.config.scaling_factor
                    else:
                        mask_latents, masked_video_latents = self.prepare_mask_latents(
                            mask_condition_tile,
                            masked_video,
                            batch_size,
                            height,
                            width,
                            dtype,
                            device,
                            generator,
                            self.do_classifier_free_guidance,
                            noise_aug_strength=noise_aug_strength,
                        )
                    
                    mask_input = torch.cat([mask_latents] * 2) if self.do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if self.do_classifier_free_guidance else masked_video_latents
                    )
                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(dtype)
                else:
                    inpaint_latents = None

                mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(device, dtype)
        else:
            if num_channels_transformer != num_channels_latents:
                mask = torch.zeros_like(latents).to(device, dtype)
                masked_video_latents = torch.zeros_like(latents).to(device, dtype)

                mask_input = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if self.do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(dtype)
            else:
                mask = torch.zeros_like(init_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(device, dtype)

                inpaint_latents = None
        if comfyui_progressbar:
            pbar.update(1)

        # Check that sizes of mask, masked image and latents match
        if num_channels_transformer != num_channels_latents:
            num_channels_mask = mask_latents.shape[1]
            num_channels_masked_image = masked_video_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.transformer.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.transformer`: {self.transformer.config} expects"
                    f" {self.transformer.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.transformer` or your `mask_image` or `image` input."
                )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9 create image_rotary_emb, style embedding & time ids
        grid_height = height // 8 // self.transformer.config.patch_size
        grid_width = width // 8 // self.transformer.config.patch_size
        if self.transformer.config.get("time_position_encoding_type", "2d_rope") == "3d_rope":
            base_size_width = 720 // 8 // self.transformer.config.patch_size
            base_size_height = 480 // 8 // self.transformer.config.patch_size

            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            image_rotary_emb = get_3d_rotary_pos_embed(
                self.transformer.config.attention_head_dim, grid_crops_coords, grid_size=(grid_height, grid_width),
                temporal_size=latents.size(2), use_real=True,
            )
        else:
            base_size = 512 // 8 // self.transformer.config.patch_size
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size, base_size
            )
            image_rotary_emb = get_2d_rotary_pos_embed(
                self.transformer.config.attention_head_dim, grid_crops_coords, (grid_height, grid_width)
            )

        # Get other hunyuan params
        style = torch.tensor([0], device=device)

        target_size = target_size or (height, width)
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            style = torch.cat([style] * 2, dim=0)

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device)
        prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
        add_time_ids = add_time_ids.to(dtype=dtype, device=device).repeat(
            batch_size * num_images_per_prompt, 1
        )
        style = style.to(device=device).repeat(batch_size * num_images_per_prompt)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if i < len(timesteps) * (1 - clip_apply_ratio) and clip_encoder_hidden_states_input is not None:
                    clip_encoder_hidden_states_actual_input = torch.zeros_like(clip_encoder_hidden_states_input)
                    clip_attention_mask_actual_input = torch.zeros_like(clip_attention_mask_input)
                else:
                    clip_encoder_hidden_states_actual_input = clip_encoder_hidden_states_input
                    clip_attention_mask_actual_input = clip_attention_mask_input
                
                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=device).to(
                    dtype=latent_model_input.dtype
                )

                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    t_expand,
                    encoder_hidden_states=prompt_embeds,
                    text_embedding_mask=prompt_attention_mask,
                    encoder_hidden_states_t5=prompt_embeds_2,
                    text_embedding_mask_t5=prompt_attention_mask_2,
                    image_meta_size=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    inpaint_latents=inpaint_latents,
                    clip_encoder_hidden_states=clip_encoder_hidden_states_actual_input,
                    clip_attention_mask=clip_attention_mask_actual_input,
                    return_dict=False,
                )[0]
                if noise_pred.size()[1] != self.vae.config.latent_channels:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_transformer == 4:
                    init_latents_proper = image_latents
                    init_mask = mask
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )
                    
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    prompt_embeds_2 = callback_outputs.pop("prompt_embeds_2", prompt_embeds_2)
                    negative_prompt_embeds_2 = callback_outputs.pop(
                        "negative_prompt_embeds_2", negative_prompt_embeds_2
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

                if comfyui_progressbar:
                    pbar.update(1)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "latent":
            video = torch.from_numpy(video)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return video

        return EasyAnimatePipelineOutput(videos=video)