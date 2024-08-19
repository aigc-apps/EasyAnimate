"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/app.py
"""
import base64
import gc
import json
import os
import random
import cv2
from datetime import datetime
from glob import glob

import gradio as gr
import numpy as np
import pkg_resources
import requests
import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                          T5EncoderModel, T5Tokenizer)

from easyanimate.data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import (HunyuanTransformer3DModel,
                                              Transformer3DModel)
from easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from easyanimate.pipeline.pipeline_easyanimate_inpaint import \
    EasyAnimateInpaintPipeline
from easyanimate.pipeline.pipeline_easyanimate_multi_text_encoder import \
    EasyAnimatePipeline_Multi_Text_Encoder
from easyanimate.pipeline.pipeline_easyanimate_multi_text_encoder_inpaint import \
    EasyAnimatePipeline_Multi_Text_Encoder_Inpaint
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import (
    get_image_to_video_latent,
    get_video_to_video_latent,
    get_width_and_height_from_image_and_base_resolution, save_videos_grid)

scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}

gradio_version = pkg_resources.get_distribution("gradio").version
gradio_version_is_above_4 = True if int(gradio_version.split('.')[0]) >= 4 else False

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

class EasyAnimateController:
    def __init__(self, low_gpu_memory_mode, weight_dtype):
        # config dirs
        self.basedir                    = os.getcwd()
        self.config_dir                 = os.path.join(self.basedir, "config")
        self.diffusion_transformer_dir  = os.path.join(self.basedir, "models", "Diffusion_Transformer")
        self.motion_module_dir          = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        self.savedir                    = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample             = os.path.join(self.savedir, "sample")
        self.edition                    = "v4"
        self.inference_config           = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_slicevae_multi_text_encoder_v4.yaml"))
        os.makedirs(self.savedir, exist_ok=True)

        self.diffusion_transformer_list = []
        self.motion_module_list      = []
        self.personalized_model_list = []
        
        self.refresh_diffusion_transformer()
        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.transformer           = None
        self.pipeline              = None
        self.motion_module_path    = "none"
        self.base_model_path       = "none"
        self.lora_model_path       = "none"
        self.low_gpu_memory_mode   = low_gpu_memory_mode
        
        self.weight_dtype = weight_dtype

    def refresh_diffusion_transformer(self):
        self.diffusion_transformer_list = sorted(glob(os.path.join(self.diffusion_transformer_dir, "*/")))

    def refresh_motion_module(self):
        motion_module_list = sorted(glob(os.path.join(self.motion_module_dir, "*.safetensors")))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = sorted(glob(os.path.join(self.personalized_model_dir, "*.safetensors")))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]
    
    def update_edition(self, edition):
        print("Update edition of EasyAnimate")
        self.edition = edition
        if edition == "v1":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_motion_module_v1.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=True), gr.update(visible=True), \
                gr.update(value=512, minimum=384, maximum=704, step=32), \
                gr.update(value=512, minimum=384, maximum=704, step=32), gr.update(value=80, minimum=40, maximum=80, step=1)
        elif edition == "v2":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_magvit_motion_module_v2.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(value=672, minimum=128, maximum=1280, step=16), \
                gr.update(value=384, minimum=128, maximum=1280, step=16), gr.update(value=144, minimum=9, maximum=144, step=9)
        elif edition == "v3":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_slicevae_motion_module_v3.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(value=672, minimum=128, maximum=1280, step=16), \
                gr.update(value=384, minimum=128, maximum=1280, step=16), gr.update(value=144, minimum=8, maximum=144, step=8)
        else:
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_slicevae_multi_text_encoder_v4.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(value=672, minimum=128, maximum=1344, step=16), \
                gr.update(value=384, minimum=128, maximum=1344, step=16), gr.update(value=144, minimum=8, maximum=144, step=8)

    def update_diffusion_transformer(self, diffusion_transformer_dropdown):
        print("Update diffusion transformer")
        if diffusion_transformer_dropdown == "none":
            return gr.update()
        if OmegaConf.to_container(self.inference_config['vae_kwargs'])['enable_magvit']:
            Choosen_AutoencoderKL = AutoencoderKLMagvit
        else:
            Choosen_AutoencoderKL = AutoencoderKL
        self.vae = Choosen_AutoencoderKL.from_pretrained(
            diffusion_transformer_dropdown, 
            subfolder="vae", 
        ).to(self.weight_dtype)
        if OmegaConf.to_container(self.inference_config['vae_kwargs'])['enable_magvit'] and self.weight_dtype == torch.float16:
            self.vae.upcast_vae = True
            
        transformer_additional_kwargs = OmegaConf.to_container(self.inference_config['transformer_additional_kwargs'])
        if self.weight_dtype == torch.float16:
            transformer_additional_kwargs["upcast_attention"] = True

        # Get Transformer
        if self.inference_config.get('enable_multi_text_encoder', False):
            Choosen_Transformer3DModel = HunyuanTransformer3DModel
        else:
            Choosen_Transformer3DModel = Transformer3DModel
        self.transformer = Choosen_Transformer3DModel.from_pretrained_2d(
            diffusion_transformer_dropdown, 
            subfolder="transformer", 
            transformer_additional_kwargs=transformer_additional_kwargs
        ).to(self.weight_dtype)
        
        # Get pipeline
        if self.inference_config.get('enable_multi_text_encoder', False):
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="image_encoder"
                ).to("cuda", self.weight_dtype)
                clip_image_processor = CLIPImageProcessor.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="image_encoder"
                )
                self.pipeline = EasyAnimatePipeline_Multi_Text_Encoder_Inpaint.from_pretrained(
                    diffusion_transformer_dropdown,
                    vae=self.vae, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"].from_pretrained(diffusion_transformer_dropdown, subfolder="scheduler"),
                    clip_image_encoder=clip_image_encoder,
                    clip_image_processor=clip_image_processor,
                )
            else:
                self.pipeline = EasyAnimatePipeline_Multi_Text_Encoder.from_pretrained(
                    diffusion_transformer_dropdown,
                    vae=self.vae, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"].from_pretrained(diffusion_transformer_dropdown, subfolder="scheduler"),
                )
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(diffusion_transformer_dropdown, subfolder="tokenizer")
            self.text_encoder = T5EncoderModel.from_pretrained(diffusion_transformer_dropdown, subfolder="text_encoder", torch_dtype=self.weight_dtype)

            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="image_encoder"
                ).to("cuda", self.weight_dtype)
                clip_image_processor = CLIPImageProcessor.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="image_encoder"
                )
                self.pipeline = EasyAnimateInpaintPipeline(
                    vae=self.vae, 
                    text_encoder=self.text_encoder, 
                    tokenizer=self.tokenizer, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)),
                    clip_image_encoder=clip_image_encoder,
                    clip_image_processor=clip_image_processor,
                )
            else:
                self.pipeline = EasyAnimatePipeline(
                    vae=self.vae, 
                    text_encoder=self.text_encoder, 
                    tokenizer=self.tokenizer, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
                )
        if self.low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()
        print("Update diffusion transformer done")
        return gr.update()

    def update_motion_module(self, motion_module_dropdown):
        self.motion_module_path = motion_module_dropdown
        print("Update motion module")
        if motion_module_dropdown == "none":
            return gr.update()
        if self.transformer is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
            if motion_module_dropdown.endswith(".safetensors"):
                from safetensors.torch import load_file, safe_open
                motion_module_state_dict = load_file(motion_module_dropdown)
            else:
                if not os.path.isfile(motion_module_dropdown):
                    raise RuntimeError(f"{motion_module_dropdown} does not exist")
                motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
            missing, unexpected = self.transformer.load_state_dict(motion_module_state_dict, strict=False)
            print("Update motion module done.")
            return gr.update()

    def update_base_model(self, base_model_dropdown):
        self.base_model_path = base_model_dropdown
        print("Update base model")
        if base_model_dropdown == "none":
            return gr.update()
        if self.transformer is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)
            self.transformer.load_state_dict(base_model_state_dict, strict=False)
            print("Update base done")
            return gr.update()

    def update_lora_model(self, lora_model_dropdown):
        print("Update lora model")
        if lora_model_dropdown == "none":
            self.lora_model_path = "none"
            return gr.update()
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_path = lora_model_dropdown
        return gr.update()

    def generate(
        self,
        diffusion_transformer_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        overlap_video_length, 
        partial_video_length, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video,
        denoise_strength,
        seed_textbox,
        is_api = False,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if self.transformer is None:
            raise gr.Error(f"Please select a pretrained model path.")

        if self.base_model_path != base_model_dropdown:
            self.update_base_model(base_model_dropdown)

        if self.motion_module_path != motion_module_dropdown:
            self.update_motion_module(motion_module_dropdown)

        if self.lora_model_path != lora_model_dropdown:
            print("Update lora model")
            self.update_lora_model(lora_model_dropdown)
        
        if resize_method == "Resize according to Reference":
            if start_image is None and validation_video is None:
                if is_api:
                    return "", f"Please upload an image when using \"Resize according to Reference\"."
                else:
                    raise gr.Error(f"Please upload an image when using \"Resize according to Reference\".")

            aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            
            if validation_video is not None:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(validation_video).read()[1]).size
            else:
                original_width, original_height = start_image[0].size if type(start_image) is list else Image.open(start_image).size
            closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
            height_slider, width_slider = [int(x / 16) * 16 for x in closest_size]

        if self.transformer.config.in_channels == self.vae.config.latent_channels and start_image is not None:
            if is_api:
                return "", f"Please select an image to video pretrained model while using image to video."
            else:
                raise gr.Error(f"Please select an image to video pretrained model while using image to video.")

        if self.transformer.config.in_channels == self.vae.config.latent_channels and generation_method == "Long Video Generation":
            if is_api:
                return "", f"Please select an image to video pretrained model while using long video generation."
            else:
                raise gr.Error(f"Please select an image to video pretrained model while using long video generation.")
        
        if start_image is None and end_image is not None:
            if is_api:
                return "", f"If specifying the ending image of the video, please specify a starting image of the video."
            else:
                raise gr.Error(f"If specifying the ending image of the video, please specify a starting image of the video.")

        is_image = True if generation_method == "Image Generation" else False

        if is_xformers_available() and not self.inference_config.get('enable_multi_text_encoder', False): self.transformer.enable_xformers_memory_efficient_attention()

        self.pipeline.scheduler = scheduler_dict[sampler_dropdown].from_config(self.pipeline.scheduler.config)
        if self.lora_model_path != "none":
            # lora part
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)

        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device="cuda").manual_seed(int(seed_textbox))
        
        try:
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                if generation_method == "Long Video Generation":
                    if validation_video is not None:
                        raise gr.Error(f"Video to Video is not Support Long Video Generation now.")
                    init_frames = 0
                    last_frames = init_frames + partial_video_length
                    while init_frames < length_slider:
                        if last_frames >= length_slider:
                            if self.pipeline.vae.quant_conv.weight.ndim==5:
                                mini_batch_encoder = self.pipeline.vae.mini_batch_encoder
                                _partial_video_length = length_slider - init_frames
                                _partial_video_length = int(_partial_video_length // mini_batch_encoder * mini_batch_encoder)
                            else:
                                _partial_video_length = length_slider - init_frames
                            
                            if _partial_video_length <= 0:
                                break
                        else:
                            _partial_video_length = partial_video_length

                        if last_frames >= length_slider:
                            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, video_length=_partial_video_length, sample_size=(height_slider, width_slider))
                        else:
                            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, None, video_length=_partial_video_length, sample_size=(height_slider, width_slider))

                        with torch.no_grad():
                            sample = self.pipeline(
                                prompt_textbox, 
                                negative_prompt     = negative_prompt_textbox,
                                num_inference_steps = sample_step_slider,
                                guidance_scale      = cfg_scale_slider,
                                width               = width_slider,
                                height              = height_slider,
                                video_length        = _partial_video_length,
                                generator           = generator,

                                video        = input_video,
                                mask_video   = input_video_mask,
                                clip_image   = clip_image, 
                                strength     = 1,
                            ).videos
                        
                        if init_frames != 0:
                            mix_ratio = torch.from_numpy(
                                np.array([float(_index) / float(overlap_video_length) for _index in range(overlap_video_length)], np.float32)
                            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            
                            new_sample[:, :, -overlap_video_length:] = new_sample[:, :, -overlap_video_length:] * (1 - mix_ratio) + \
                                sample[:, :, :overlap_video_length] * mix_ratio
                            new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim = 2)

                            sample = new_sample
                        else:
                            new_sample = sample

                        if last_frames >= length_slider:
                            break

                        start_image = [
                            Image.fromarray(
                                (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
                            ) for _index in range(-overlap_video_length, 0)
                        ]

                        init_frames = init_frames + _partial_video_length - overlap_video_length
                        last_frames = init_frames + _partial_video_length
                else:
                    if validation_video is not None:
                        input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                        strength = denoise_strength
                    else:
                        input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                        strength = 1

                    sample = self.pipeline(
                        prompt_textbox,
                        negative_prompt     = negative_prompt_textbox,
                        num_inference_steps = sample_step_slider,
                        guidance_scale      = cfg_scale_slider,
                        width               = width_slider,
                        height              = height_slider,
                        video_length        = length_slider if not is_image else 1,
                        generator           = generator,

                        video        = input_video,
                        mask_video   = input_video_mask,
                        clip_image   = clip_image, 
                        strength     = strength,
                    ).videos
            else:
                sample = self.pipeline(
                    prompt_textbox,
                    negative_prompt     = negative_prompt_textbox,
                    num_inference_steps = sample_step_slider,
                    guidance_scale      = cfg_scale_slider,
                    width               = width_slider,
                    height              = height_slider,
                    video_length        = length_slider if not is_image else 1,
                    generator           = generator
                ).videos
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if self.lora_model_path != "none":
                self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            if is_api:
                return "", f"Error. error information is {str(e)}"
            else:
                return gr.update(), gr.update(), f"Error. error information is {str(e)}"

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # lora part
        if self.lora_model_path != "none":
            self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)

        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed_textbox": seed_textbox
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        if not os.path.exists(self.savedir_sample):
            os.makedirs(self.savedir_sample, exist_ok=True)
        index = len([path for path in os.listdir(self.savedir_sample)]) + 1
        prefix = str(index).zfill(3)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if is_image or length_slider == 1:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(save_sample_path)

            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
                else:
                    return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            save_videos_grid(sample, save_sample_path, fps=12 if self.edition == "v1" else 24)

            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
                else:
                    return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"


def ui(low_gpu_memory_mode, weight_dtype):
    controller = EasyAnimateController(low_gpu_memory_mode, weight_dtype)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # EasyAnimate: An End-to-End Solution for High-Resolution and Long Video Generation 
             
            Generate your videos easily.   

            EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata.   
            
            [Github](https://github.com/aigc-apps/EasyAnimate/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. EasyAnimate Edition (EasyAnimate版本).
                """
            )
            with gr.Row():
                easyanimate_edition_dropdown = gr.Dropdown(
                    label="The config of EasyAnimate Edition (EasyAnimate版本配置)",
                    choices=["v1", "v2", "v3", "v4"],
                    value="v4",
                    interactive=True,
                )
            gr.Markdown(
                """
                ### 2. Model checkpoints (模型路径).
                """
            )
            with gr.Row():
                diffusion_transformer_dropdown = gr.Dropdown(
                    label="Pretrained Model Path (预训练模型路径)",
                    choices=controller.diffusion_transformer_list,
                    value="none",
                    interactive=True,
                )
                diffusion_transformer_dropdown.change(
                    fn=controller.update_diffusion_transformer, 
                    inputs=[diffusion_transformer_dropdown], 
                    outputs=[diffusion_transformer_dropdown]
                )
                
                diffusion_transformer_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def refresh_diffusion_transformer():
                    controller.refresh_diffusion_transformer()
                    return gr.update(choices=controller.diffusion_transformer_list)
                diffusion_transformer_refresh_button.click(fn=refresh_diffusion_transformer, inputs=[], outputs=[diffusion_transformer_dropdown])

            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module (选择运动模块[非必需])",
                    choices=controller.motion_module_list,
                    value="none",
                    interactive=True,
                    visible=False
                )

                motion_module_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton", visible=False)
                def update_motion_module():
                    controller.refresh_motion_module()
                    return gr.update(choices=controller.motion_module_list)
                motion_module_refresh_button.click(fn=update_motion_module, inputs=[], outputs=[motion_module_dropdown])
                
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (选择基模型[非必需])",
                    choices=controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )
                
                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (选择LoRA模型[非必需])",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )

                lora_alpha_slider = gr.Slider(label="LoRA alpha (LoRA权重)", value=0.55, minimum=0, maximum=2, interactive=True)
                
                personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.update(choices=controller.personalized_model_list),
                        gr.update(choices=["none"] + controller.personalized_model_list)
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[base_model_dropdown, lora_model_dropdown])

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 3. Configs for Generation (生成参数配置).
                """
            )
            
            prompt_textbox = gr.Textbox(label="Prompt (正向提示词)", lines=2, value="A young woman with beautiful and clear eyes and blonde hair standing and white dress in a forest wearing a crown. She seems to be lost in thought, and the camera focuses on her face. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.")
            negative_prompt_textbox = gr.Textbox(label="Negative prompt (负向提示词)", lines=2, value="The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion." )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method (采样器种类)", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps (生成步数)", value=30, minimum=10, maximum=100, step=1)
                        
                    resize_method = gr.Radio(
                        ["Generate by", "Resize according to Reference"],
                        value="Generate by",
                        show_label=False,
                    )
                    width_slider     = gr.Slider(label="Width (视频宽度)",            value=672, minimum=128, maximum=1344, step=16)
                    height_slider    = gr.Slider(label="Height (视频高度)",           value=384, minimum=128, maximum=1344, step=16)
                    base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 768, 960], visible=False)

                    with gr.Group():
                        generation_method = gr.Radio(
                            ["Video Generation", "Image Generation", "Long Video Generation"],
                            value="Video Generation",
                            show_label=False,
                        )
                        with gr.Row():
                            length_slider = gr.Slider(label="Animation length (视频帧数)", value=144, minimum=8,   maximum=144,  step=8)
                            overlap_video_length = gr.Slider(label="Overlap length (视频续写的重叠帧数)", value=4, minimum=1,   maximum=4,  step=1, visible=False)
                            partial_video_length = gr.Slider(label="Partial video generation length (每个部分的视频生成帧数)", value=72, minimum=8,   maximum=144,  step=8, visible=False)
                    
                    source_method = gr.Radio(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)"],
                        value="Text to Video (文本到视频)",
                        show_label=False,
                    )
                    with gr.Column(visible = False) as image_to_video_col:
                        start_image = gr.Image(
                            label="The image at the beginning of the video (图片到视频的开始图片)",  show_label=True, 
                            elem_id="i2v_start", sources="upload", type="filepath", 
                        )
                        
                        template_gallery_path = ["asset/1.png", "asset/2.png", "asset/3.png", "asset/4.png", "asset/5.png"]
                        def select_template(evt: gr.SelectData):
                            text = {
                                "asset/1.png": "The dog is looking at camera and smiling. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                "asset/2.png": "a sailboat sailing in rough seas with a dramatic sunset. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                "asset/3.png": "a beautiful woman with long hair and a dress blowing in the wind. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                "asset/4.png": "a man in an astronaut suit playing a guitar. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                "asset/5.png": "fireworks display over night city. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                            }[template_gallery_path[evt.index]]
                            return template_gallery_path[evt.index], text

                        template_gallery = gr.Gallery(
                            template_gallery_path,
                            columns=5, rows=1,
                            height=140,
                            allow_preview=False,
                            container=False,
                            label="Template Examples",
                        )
                        template_gallery.select(select_template, None, [start_image, prompt_textbox])
                        
                        with gr.Accordion("The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", open=False):
                            end_image   = gr.Image(label="The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", show_label=False, elem_id="i2v_end", sources="upload", type="filepath")

                    with gr.Column(visible = False) as video_to_video_col:
                        validation_video = gr.Video(
                            label="The video to convert (视频转视频的参考视频)",  show_label=True, 
                            elem_id="v2v", sources="upload", 
                        )
                        denoise_strength = gr.Slider(label="Denoise strength (重绘系数)", value=0.70, minimum=0.10, maximum=0.95, step=0.01)

                    cfg_scale_slider  = gr.Slider(label="CFG Scale (引导系数)",        value=7.0, minimum=0,   maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed (随机种子)", value=43)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(
                            fn=lambda: gr.Textbox(value=random.randint(1, 1e8)) if gradio_version_is_above_4 else gr.Textbox.update(value=random.randint(1, 1e8)), 
                            inputs=[], 
                            outputs=[seed_textbox]
                        )

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')
                    
                with gr.Column():
                    result_image = gr.Image(label="Generated Image (生成图片)", interactive=False, visible=False)
                    result_video = gr.Video(label="Generated Animation (生成视频)", interactive=False)
                    infer_progress = gr.Textbox(
                        label="Generation Info (生成信息)",
                        value="No task currently",
                        interactive=False
                    )

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return [gr.update(visible=True, maximum=144, value=144), gr.update(visible=False), gr.update(visible=False)]
                elif generation_method == "Image Generation":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
                else:
                    return [gr.update(visible=True, maximum=1440), gr.update(visible=True), gr.update(visible=True)]
            generation_method.change(
                upload_generation_method, generation_method, [length_slider, overlap_video_length, partial_video_length]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update()]
            source_method.change(
                upload_source_method, source_method, [image_to_video_col, video_to_video_col, start_image, end_image, validation_video]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            easyanimate_edition_dropdown.change(
                fn=controller.update_edition, 
                inputs=[easyanimate_edition_dropdown], 
                outputs=[
                    easyanimate_edition_dropdown, 
                    diffusion_transformer_dropdown, 
                    motion_module_dropdown, 
                    motion_module_refresh_button, 
                    width_slider, 
                    height_slider, 
                    length_slider, 
                ]
            )
            generate_button.click(
                fn=controller.generate,
                inputs=[
                    diffusion_transformer_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_model_dropdown,
                    lora_alpha_slider,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    resize_method,
                    width_slider, 
                    height_slider, 
                    base_resolution, 
                    generation_method, 
                    length_slider, 
                    overlap_video_length, 
                    partial_video_length, 
                    cfg_scale_slider, 
                    start_image, 
                    end_image, 
                    validation_video,
                    denoise_strength, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller


class EasyAnimateController_Modelscope:
    def __init__(self, edition, config_path, model_name, savedir_sample, low_gpu_memory_mode, weight_dtype):
        # Basic dir
        self.basedir                    = os.getcwd()
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        self.lora_model_path            = "none"
        self.savedir_sample             = savedir_sample
        self.refresh_personalized_model()
        os.makedirs(self.savedir_sample, exist_ok=True)

        # Config and model path
        self.edition = edition
        self.weight_dtype = weight_dtype
        self.inference_config = OmegaConf.load(config_path)
        # Get Transformer
        transformer_additional_kwargs = OmegaConf.to_container(self.inference_config['transformer_additional_kwargs'])
        if self.weight_dtype == torch.float16:
            transformer_additional_kwargs["upcast_attention"] = True
        if self.inference_config.get('enable_multi_text_encoder', False):
            Choosen_Transformer3DModel = HunyuanTransformer3DModel
        else:
            Choosen_Transformer3DModel = Transformer3DModel
        self.transformer = Choosen_Transformer3DModel.from_pretrained_2d(
            model_name, 
            subfolder="transformer", 
            transformer_additional_kwargs=transformer_additional_kwargs
        ).to(self.weight_dtype)
        
        if OmegaConf.to_container(self.inference_config['vae_kwargs'])['enable_magvit']:
            Choosen_AutoencoderKL = AutoencoderKLMagvit
        else:
            Choosen_AutoencoderKL = AutoencoderKL
        self.vae = Choosen_AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae", 
        ).to(self.weight_dtype)
        if OmegaConf.to_container(self.inference_config['vae_kwargs'])['enable_magvit'] and self.weight_dtype == torch.float16:
            self.vae.upcast_vae = True

        # Get pipeline
        if self.inference_config.get('enable_multi_text_encoder', False):
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    model_name, subfolder="image_encoder"
                ).to("cuda", self.weight_dtype)
                clip_image_processor = CLIPImageProcessor.from_pretrained(
                    model_name, subfolder="image_encoder"
                )
                self.pipeline = EasyAnimatePipeline_Multi_Text_Encoder_Inpaint.from_pretrained(
                    model_name,
                    vae=self.vae, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"].from_pretrained(model_name, subfolder="scheduler"),
                    clip_image_encoder=clip_image_encoder,
                    clip_image_processor=clip_image_processor,
                )
            else:
                self.pipeline = EasyAnimatePipeline_Multi_Text_Encoder.from_pretrained(
                    model_name,
                    vae=self.vae, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"].from_pretrained(model_name, subfolder="scheduler"),
                )
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
            self.text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=self.weight_dtype)

            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    model_name, subfolder="image_encoder"
                ).to("cuda", self.weight_dtype)
                clip_image_processor = CLIPImageProcessor.from_pretrained(
                    model_name, subfolder="image_encoder"
                )
                self.pipeline = EasyAnimateInpaintPipeline(
                    vae=self.vae, 
                    text_encoder=self.text_encoder, 
                    tokenizer=self.tokenizer, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)),
                    clip_image_encoder=clip_image_encoder,
                    clip_image_processor=clip_image_processor,
                )
            else:
                self.pipeline = EasyAnimatePipeline(
                    vae=self.vae, 
                    text_encoder=self.text_encoder, 
                    tokenizer=self.tokenizer, 
                    transformer=self.transformer,
                    scheduler=scheduler_dict["Euler"](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
                )
        if low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()
        print("Update diffusion transformer done")


    def refresh_personalized_model(self):
        personalized_model_list = sorted(glob(os.path.join(self.personalized_model_dir, "*.safetensors")))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]


    def update_lora_model(self, lora_model_dropdown):
        print("Update lora model")
        if lora_model_dropdown == "none":
            self.lora_model_path = "none"
            return gr.update()
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_path = lora_model_dropdown
        return gr.update()

    
    def generate(
        self,
        diffusion_transformer_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        overlap_video_length, 
        partial_video_length, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video,
        denoise_strength,
        seed_textbox,
        is_api = False,
    ):    
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if self.transformer is None:
            raise gr.Error(f"Please select a pretrained model path.")

        if self.lora_model_path != lora_model_dropdown:
            print("Update lora model")
            self.update_lora_model(lora_model_dropdown)

        if resize_method == "Resize according to Reference":
            if start_image is None and validation_video is None:
                raise gr.Error(f"Please upload an image when using \"Resize according to Reference\".")
        
            if validation_video is not None:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(validation_video).read()[1]).size
            else:
                original_width, original_height = start_image[0].size if type(start_image) is list else Image.open(start_image).size
            closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
            height_slider, width_slider = [int(x / 16) * 16 for x in closest_size]

        if self.transformer.config.in_channels == self.vae.config.latent_channels and start_image is not None:
            raise gr.Error(f"Please select an image to video pretrained model while using image to video.")
        
        if start_image is None and end_image is not None:
            raise gr.Error(f"If specifying the ending image of the video, please specify a starting image of the video.")

        is_image = True if generation_method == "Image Generation" else False

        if is_xformers_available() and not self.inference_config.get('enable_multi_text_encoder', False): self.transformer.enable_xformers_memory_efficient_attention()

        self.pipeline.scheduler = scheduler_dict[sampler_dropdown].from_config(self.pipeline.scheduler.config)
        if self.lora_model_path != "none":
            # lora part
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)

        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device="cuda").manual_seed(int(seed_textbox))
        
        try:
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                if validation_video is not None:
                    input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                    strength = denoise_strength
                else:
                    input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                    strength = 1

                sample = self.pipeline(
                    prompt_textbox,
                    negative_prompt     = negative_prompt_textbox,
                    num_inference_steps = sample_step_slider,
                    guidance_scale      = cfg_scale_slider,
                    width               = width_slider,
                    height              = height_slider,
                    video_length        = length_slider if not is_image else 1,
                    generator           = generator,

                    video        = input_video,
                    mask_video   = input_video_mask,
                    clip_image   = clip_image, 
                    strength     = strength,
                ).videos
            else:
                sample = self.pipeline(
                    prompt_textbox,
                    negative_prompt     = negative_prompt_textbox,
                    num_inference_steps = sample_step_slider,
                    guidance_scale      = cfg_scale_slider,
                    width               = width_slider,
                    height              = height_slider,
                    video_length        = length_slider if not is_image else 1,
                    generator           = generator
                ).videos
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if self.lora_model_path != "none":
                self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            if is_api:
                return "", f"Error. error information is {str(e)}"
            else:
                return gr.update(), gr.update(), f"Error. error information is {str(e)}"

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # lora part
        if self.lora_model_path != "none":
            self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)

        if not os.path.exists(self.savedir_sample):
            os.makedirs(self.savedir_sample, exist_ok=True)
        index = len([path for path in os.listdir(self.savedir_sample)]) + 1
        prefix = str(index).zfill(3)
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if is_image or length_slider == 1:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(save_sample_path)
            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
                else:
                    return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            save_videos_grid(sample, save_sample_path, fps=12 if self.edition == "v1" else 24)
            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
                else:
                    return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"


def ui_modelscope(edition, config_path, model_name, savedir_sample, low_gpu_memory_mode, weight_dtype):
    controller = EasyAnimateController_Modelscope(edition, config_path, model_name, savedir_sample, low_gpu_memory_mode, weight_dtype)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # EasyAnimate: An End-to-End Solution for High-Resolution and Long Video Generation 
             
            Generate your videos easily.   

            EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata.   
            
            [Github](https://github.com/aigc-apps/EasyAnimate/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model checkpoints (模型路径).
                """
            )
            with gr.Row():
                diffusion_transformer_dropdown = gr.Dropdown(
                    label="Pretrained Model Path (预训练模型路径)",
                    choices=[model_name],
                    value=model_name,
                    interactive=False,
                )
            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module (选择运动模块[非必需])",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (选择基模型[非必需])",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                with gr.Column(visible=False):
                    gr.Markdown(
                        """
                        ### Minimalism is an example portrait of Lora, triggered by specific prompt words. More details can be found on [Wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-Lora).
                        """
                    )
                    with gr.Row():
                        lora_model_dropdown = gr.Dropdown(
                            label="Select LoRA model",
                            choices=["none", "easyanimatev2_minimalism_lora.safetensors"],
                            value="none",
                            interactive=True,
                        )

                        lora_alpha_slider = gr.Slider(label="LoRA alpha (LoRA权重)", value=0.55, minimum=0, maximum=2, interactive=True)
                
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for Generation (生成参数配置).
                """
            )

            prompt_textbox = gr.Textbox(label="Prompt (正向提示词)", lines=2, value="A young woman with beautiful and clear eyes and blonde hair standing and white dress in a forest wearing a crown. She seems to be lost in thought, and the camera focuses on her face. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.")
            negative_prompt_textbox = gr.Textbox(label="Negative prompt (负向提示词)", lines=2, value="The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion." )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method (采样器种类)", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps (生成步数)", value=20, minimum=10, maximum=30, step=1, interactive=False)
                    
                    if edition == "v1":
                        width_slider     = gr.Slider(label="Width (视频宽度)",            value=512, minimum=384, maximum=704, step=32)
                        height_slider    = gr.Slider(label="Height (视频高度)",           value=512, minimum=384, maximum=704, step=32)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=False,
                            )
                            length_slider = gr.Slider(label="Animation length (视频帧数)", value=80,  minimum=40,  maximum=96,   step=1)
                        overlap_video_length = gr.Slider(label="Overlap length (视频续写的重叠帧数)", value=4, minimum=1,   maximum=4,  step=1, visible=False)
                        partial_video_length = gr.Slider(label="Partial video generation length (每个部分的视频生成帧数)", value=72, minimum=8,   maximum=144,  step=8, visible=False)
                        cfg_scale_slider = gr.Slider(label="CFG Scale (引导系数)",        value=6.0, minimum=0,   maximum=20)
                    else:
                        resize_method = gr.Radio(
                            ["Generate by", "Resize according to Reference"],
                            value="Generate by",
                            show_label=False,
                        )                        
                        with gr.Column():
                            gr.Markdown(
                                """                    
                                We support video generation up to 720p with 144 frames, but for the trial experience, we have set certain limitations. We fix the max resolution of video to 384x672x48 (2s). 

                                If the start image you uploaded does not match this resolution, you can use the "Resize according to Reference" option above. 
                                
                                If you want to experience longer and larger video generation, you can go to our [Github](https://github.com/aigc-apps/EasyAnimate/). 
                                """
                            )
                        width_slider     = gr.Slider(label="Width (视频宽度)",            value=672, minimum=128, maximum=1280, step=16, interactive=False)
                        height_slider    = gr.Slider(label="Height (视频高度)",           value=384, minimum=128, maximum=1280, step=16, interactive=False)
                        base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 768, 960], interactive=False, visible=False)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=True,
                            )
                            length_slider = gr.Slider(label="Animation length (视频帧数)", value=48, minimum=8,   maximum=48,  step=8)
                            overlap_video_length = gr.Slider(label="Overlap length (视频续写的重叠帧数)", value=4, minimum=1,   maximum=4,  step=1, visible=False)
                            partial_video_length = gr.Slider(label="Partial video generation length (每个部分的视频生成帧数)", value=72, minimum=8,   maximum=144,  step=8, visible=False)
                        
                        source_method = gr.Radio(
                            ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)"],
                            value="Text to Video (文本到视频)",
                            show_label=False,
                        )
                        with gr.Column(visible = False) as image_to_video_col:
                            with gr.Row():
                                start_image = gr.Image(label="The image at the beginning of the video (图片到视频的开始图片)", show_label=True, elem_id="i2v_start", sources="upload", type="filepath")
                            
                            template_gallery_path = ["asset/1.png", "asset/2.png", "asset/3.png", "asset/4.png", "asset/5.png"]
                            def select_template(evt: gr.SelectData):
                                text = {
                                    "asset/1.png": "The dog is looking at camera and smiling. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/2.png": "a sailboat sailing in rough seas with a dramatic sunset. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/3.png": "a beautiful woman with long hair and a dress blowing in the wind. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/4.png": "a man in an astronaut suit playing a guitar. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/5.png": "fireworks display over night city. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                }[template_gallery_path[evt.index]]
                                return template_gallery_path[evt.index], text

                            template_gallery = gr.Gallery(
                                template_gallery_path,
                                columns=5, rows=1,
                                height=140,
                                allow_preview=False,
                                container=False,
                                label="Template Examples",
                            )
                            template_gallery.select(select_template, None, [start_image, prompt_textbox])

                            with gr.Accordion("The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", open=False):
                                end_image   = gr.Image(label="The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", show_label=False, elem_id="i2v_end", sources="upload", type="filepath")

                        with gr.Column(visible = False) as video_to_video_col:
                            validation_video = gr.Video(
                                label="The video to convert (视频转视频的参考视频)",  show_label=True, 
                                elem_id="v2v", sources="upload", 
                            )
                            denoise_strength = gr.Slider(label="Denoise strength (重绘系数)", value=0.70, minimum=0.10, maximum=0.95, step=0.01)

                        cfg_scale_slider  = gr.Slider(label="CFG Scale (引导系数)",        value=7.0, minimum=0,   maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed (随机种子)", value=43)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(
                            fn=lambda: gr.Textbox(value=random.randint(1, 1e8)) if gradio_version_is_above_4 else gr.Textbox.update(value=random.randint(1, 1e8)), 
                            inputs=[], 
                            outputs=[seed_textbox]
                        )

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')
                    
                with gr.Column():
                    result_image = gr.Image(label="Generated Image (生成图片)", interactive=False, visible=False)
                    result_video = gr.Video(label="Generated Animation (生成视频)", interactive=False)
                    infer_progress = gr.Textbox(
                        label="Generation Info (生成信息)",
                        value="No task currently",
                        interactive=False
                    )

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return gr.update(visible=True, minimum=8, maximum=48, value=48, interactive=True)
                elif generation_method == "Image Generation":
                    return gr.update(minimum=1, maximum=1, value=1, interactive=False)
            generation_method.change(
                upload_generation_method, generation_method, [length_slider]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update()]
            source_method.change(
                upload_source_method, source_method, [image_to_video_col, video_to_video_col, start_image, end_image, validation_video]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
                    diffusion_transformer_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_model_dropdown, 
                    lora_alpha_slider,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    resize_method,
                    width_slider, 
                    height_slider, 
                    base_resolution, 
                    generation_method, 
                    length_slider, 
                    overlap_video_length, 
                    partial_video_length, 
                    cfg_scale_slider, 
                    start_image, 
                    end_image, 
                    validation_video,
                    denoise_strength, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller


def post_eas(
    diffusion_transformer_dropdown, motion_module_dropdown,
    base_model_dropdown, lora_model_dropdown, lora_alpha_slider,
    prompt_textbox, negative_prompt_textbox, 
    sampler_dropdown, sample_step_slider, resize_method, width_slider, height_slider,
    base_resolution, generation_method, length_slider, cfg_scale_slider, 
    start_image, end_image, validation_video, denoise_strength, seed_textbox,
):
    if start_image is not None:
        with open(start_image, 'rb') as file:
            file_content = file.read()
            start_image_encoded_content = base64.b64encode(file_content)
            start_image = start_image_encoded_content.decode('utf-8')

    if end_image is not None:
        with open(end_image, 'rb') as file:
            file_content = file.read()
            end_image_encoded_content = base64.b64encode(file_content)
            end_image = end_image_encoded_content.decode('utf-8')

    if validation_video is not None:
        with open(validation_video, 'rb') as file:
            file_content = file.read()
            validation_video_encoded_content = base64.b64encode(file_content)
            validation_video = validation_video_encoded_content.decode('utf-8')

    datas = {
        "base_model_path": base_model_dropdown,
        "motion_module_path": motion_module_dropdown,
        "lora_model_path": lora_model_dropdown, 
        "lora_alpha_slider": lora_alpha_slider, 
        "prompt_textbox": prompt_textbox, 
        "negative_prompt_textbox": negative_prompt_textbox, 
        "sampler_dropdown": sampler_dropdown, 
        "sample_step_slider": sample_step_slider, 
        "resize_method": resize_method,
        "width_slider": width_slider, 
        "height_slider": height_slider, 
        "base_resolution": base_resolution,
        "generation_method": generation_method,
        "length_slider": length_slider,
        "cfg_scale_slider": cfg_scale_slider,
        "start_image": start_image,
        "end_image": end_image,
        "validation_video": validation_video,
        "denoise_strength": denoise_strength,
        "seed_textbox": seed_textbox,
    }

    session = requests.session()
    session.headers.update({"Authorization": os.environ.get("EAS_TOKEN")})

    response = session.post(url=f'{os.environ.get("EAS_URL")}/easyanimate/infer_forward', json=datas, timeout=300)

    outputs = response.json()
    return outputs


class EasyAnimateController_EAS:
    def __init__(self, edition, config_path, model_name, savedir_sample):
        self.savedir_sample = savedir_sample
        os.makedirs(self.savedir_sample, exist_ok=True)

    def generate(
        self,
        diffusion_transformer_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video, 
        denoise_strength,
        seed_textbox
    ):
        is_image = True if generation_method == "Image Generation" else False

        outputs = post_eas(
            diffusion_transformer_dropdown, motion_module_dropdown,
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider,
            prompt_textbox, negative_prompt_textbox, 
            sampler_dropdown, sample_step_slider, resize_method, width_slider, height_slider,
            base_resolution, generation_method, length_slider, cfg_scale_slider, 
            start_image, end_image, validation_video, denoise_strength, 
            seed_textbox
        )
        try:
            base64_encoding = outputs["base64_encoding"]
        except:
            return gr.Image(visible=False, value=None), gr.Video(None, visible=True), outputs["message"]
            
        decoded_data = base64.b64decode(base64_encoding)

        if not os.path.exists(self.savedir_sample):
            os.makedirs(self.savedir_sample, exist_ok=True)
        index = len([path for path in os.listdir(self.savedir_sample)]) + 1
        prefix = str(index).zfill(3)
        
        if is_image or length_slider == 1:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")
            with open(save_sample_path, "wb") as file:
                file.write(decoded_data)
            if gradio_version_is_above_4:
                return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
            else:
                return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            with open(save_sample_path, "wb") as file:
                file.write(decoded_data)
            if gradio_version_is_above_4:
                return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
            else:
                return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"


def ui_eas(edition, config_path, model_name, savedir_sample):
    controller = EasyAnimateController_EAS(edition, config_path, model_name, savedir_sample)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # EasyAnimate: An End-to-End Solution for High-Resolution and Long Video Generation 
             
            Generate your videos easily.   

            EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata.   
            
            [Github](https://github.com/aigc-apps/EasyAnimate/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model checkpoints.
                """
            )
            with gr.Row():
                diffusion_transformer_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=[model_name],
                    value=model_name,
                    interactive=False,
                )
            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                with gr.Column(visible=False):
                    gr.Markdown(
                        """
                        ### Minimalism is an example portrait of Lora, triggered by specific prompt words. More details can be found on [Wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-Lora).
                        """
                    )
                    with gr.Row():
                        lora_model_dropdown = gr.Dropdown(
                            label="Select LoRA model",
                            choices=["none", "easyanimatev2_minimalism_lora.safetensors"],
                            value="none",
                            interactive=True,
                        )

                        lora_alpha_slider = gr.Slider(label="LoRA alpha (LoRA权重)", value=0.55, minimum=0, maximum=2, interactive=True)
                
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for Generation.
                """
            )
            
            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="A young woman with beautiful and clear eyes and blonde hair standing and white dress in a forest wearing a crown. She seems to be lost in thought, and the camera focuses on her face. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.")
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion. " )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=20, minimum=10, maximum=30, step=1, interactive=False)
                    
                    if edition == "v1":
                        width_slider     = gr.Slider(label="Width",            value=512, minimum=384, maximum=704, step=32)
                        height_slider    = gr.Slider(label="Height",           value=512, minimum=384, maximum=704, step=32)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=False,
                            )
                            length_slider    = gr.Slider(label="Animation length", value=80,  minimum=40,  maximum=96,   step=1)
                        cfg_scale_slider = gr.Slider(label="CFG Scale",        value=6.0, minimum=0,   maximum=20)
                    else:
                        resize_method = gr.Radio(
                            ["Generate by", "Resize according to Reference"],
                            value="Generate by",
                            show_label=False,
                        )                        
                        with gr.Column():
                            gr.Markdown(
                                """                    
                                We support video generation up to 720p with 144 frames, but for the trial experience, we have set certain limitations. We fix the max resolution of video to 384x672x48 (2s). 

                                If the start image you uploaded does not match this resolution, you can use the "Resize according to Reference" option above. 
                                
                                If you want to experience longer and larger video generation, you can go to our [Github](https://github.com/aigc-apps/EasyAnimate/). 
                                """
                            )
                        width_slider     = gr.Slider(label="Width (视频宽度)",            value=672, minimum=128, maximum=1280, step=16, interactive=False)
                        height_slider    = gr.Slider(label="Height (视频高度)",           value=384, minimum=128, maximum=1280, step=16, interactive=False)
                        base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 768, 960], interactive=False, visible=False)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=True,
                            )
                            length_slider = gr.Slider(label="Animation length (视频帧数)", value=48, minimum=8,   maximum=48,  step=8)
                        
                        source_method = gr.Radio(
                            ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)"],
                            value="Text to Video (文本到视频)",
                            show_label=False,
                        )
                        with gr.Column(visible = False) as image_to_video_col:
                            start_image = gr.Image(label="The image at the beginning of the video", show_label=True, elem_id="i2v_start", sources="upload", type="filepath")
                            
                            template_gallery_path = ["asset/1.png", "asset/2.png", "asset/3.png", "asset/4.png", "asset/5.png"]
                            def select_template(evt: gr.SelectData):
                                text = {
                                    "asset/1.png": "The dog is looking at camera and smiling. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/2.png": "a sailboat sailing in rough seas with a dramatic sunset. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/3.png": "a beautiful woman with long hair and a dress blowing in the wind. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/4.png": "a man in an astronaut suit playing a guitar. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                    "asset/5.png": "fireworks display over night city. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", 
                                }[template_gallery_path[evt.index]]
                                return template_gallery_path[evt.index], text

                            template_gallery = gr.Gallery(
                                template_gallery_path,
                                columns=5, rows=1,
                                height=140,
                                allow_preview=False,
                                container=False,
                                label="Template Examples",
                            )
                            template_gallery.select(select_template, None, [start_image, prompt_textbox])

                            with gr.Accordion("The image at the ending of the video (Optional)", open=False):
                                end_image   = gr.Image(label="The image at the ending of the video (Optional)", show_label=True, elem_id="i2v_end", sources="upload", type="filepath")
                        
                        with gr.Column(visible = False) as video_to_video_col:
                            validation_video = gr.Video(
                                label="The video to convert (视频转视频的参考视频)",  show_label=True, 
                                elem_id="v2v", sources="upload", 
                            )
                            denoise_strength = gr.Slider(label="Denoise strength (重绘系数)", value=0.70, minimum=0.10, maximum=0.95, step=0.01)

                        cfg_scale_slider  = gr.Slider(label="CFG Scale (引导系数)",        value=7.0, minimum=0,   maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=43)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(
                            fn=lambda: gr.Textbox(value=random.randint(1, 1e8)) if gradio_version_is_above_4 else gr.Textbox.update(value=random.randint(1, 1e8)), 
                            inputs=[], 
                            outputs=[seed_textbox]
                        )

                    generate_button = gr.Button(value="Generate", variant='primary')
                    
                with gr.Column():
                    result_image = gr.Image(label="Generated Image", interactive=False, visible=False)
                    result_video = gr.Video(label="Generated Animation", interactive=False)
                    infer_progress = gr.Textbox(
                        label="Generation Info",
                        value="No task currently",
                        interactive=False
                    )

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return gr.update(visible=True, minimum=8, maximum=48, value=48, interactive=True)
                elif generation_method == "Image Generation":
                    return gr.update(minimum=1, maximum=1, value=1, interactive=False)
            generation_method.change(
                upload_generation_method, generation_method, [length_slider]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update()]
            source_method.change(
                upload_source_method, source_method, [image_to_video_col, video_to_video_col, start_image, end_image, validation_video]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
                    diffusion_transformer_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_model_dropdown, 
                    lora_alpha_slider,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    resize_method,
                    width_slider, 
                    height_slider, 
                    base_resolution, 
                    generation_method, 
                    length_slider, 
                    cfg_scale_slider, 
                    start_image, 
                    end_image, 
                    validation_video,
                    denoise_strength, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller