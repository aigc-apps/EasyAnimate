"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/app.py
"""
import gc
import json
import os
import random
import base64
import requests
import pkg_resources
from datetime import datetime
from glob import glob

import gradio as gr
import torch
import numpy as np
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import T5EncoderModel, T5Tokenizer

from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import save_videos_grid
from PIL import Image

sample_idx = 0
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
    def __init__(self):
        # config dirs
        self.basedir                    = os.getcwd()
        self.config_dir                 = os.path.join(self.basedir, "config")
        self.diffusion_transformer_dir  = os.path.join(self.basedir, "models", "Diffusion_Transformer")
        self.motion_module_dir          = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        self.savedir                    = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample             = os.path.join(self.savedir, "sample")
        self.edition                    = "v2"
        self.inference_config           = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_magvit_motion_module_v2.yaml"))
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
        
        self.weight_dtype = torch.bfloat16

    def refresh_diffusion_transformer(self):
        self.diffusion_transformer_list = glob(os.path.join(self.diffusion_transformer_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.safetensors"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]
    
    def update_edition(self, edition):
        print("Update edition of EasyAnimate")
        self.edition = edition
        if edition == "v1":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_motion_module_v1.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=True), gr.update(visible=True), \
                gr.update(visible=False), gr.update(value=512, minimum=384, maximum=704, step=32), \
                gr.update(value=512, minimum=384, maximum=704, step=32), gr.update(value=80, minimum=40, maximum=80, step=1)
        else:
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_magvit_motion_module_v2.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(visible=True), gr.update(value=672, minimum=128, maximum=1280, step=16), \
                gr.update(value=384, minimum=128, maximum=1280, step=16), gr.update(value=144, minimum=9, maximum=144, step=9)

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
        self.transformer = Transformer3DModel.from_pretrained_2d(
            diffusion_transformer_dropdown, 
            subfolder="transformer", 
            transformer_additional_kwargs=OmegaConf.to_container(self.inference_config.transformer_additional_kwargs)
        ).to(self.weight_dtype)
        self.tokenizer = T5Tokenizer.from_pretrained(diffusion_transformer_dropdown, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(diffusion_transformer_dropdown, subfolder="text_encoder", torch_dtype=self.weight_dtype)

        # Get pipeline
        self.pipeline = EasyAnimatePipeline(
            vae=self.vae, 
            text_encoder=self.text_encoder, 
            tokenizer=self.tokenizer, 
            transformer=self.transformer,
            scheduler=scheduler_dict["Euler"](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        )
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
        width_slider, 
        height_slider, 
        is_image,
        length_slider, 
        cfg_scale_slider, 
        seed_textbox,
        is_api = False,
    ):
        global sample_idx
        if self.transformer is None:
            raise gr.Error(f"Please select a pretrained model path.")

        if self.base_model_path != base_model_dropdown:
            self.update_base_model(base_model_dropdown)

        if self.motion_module_path != motion_module_dropdown:
            self.update_motion_module(motion_module_dropdown)

        if self.lora_model_path != lora_model_dropdown:
            print("Update lora model")
            self.update_lora_model(lora_model_dropdown)

        if is_xformers_available(): self.transformer.enable_xformers_memory_efficient_attention()

        self.pipeline.scheduler = scheduler_dict[sampler_dropdown](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        if self.lora_model_path != "none":
            # lora part
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
        self.pipeline.to("cuda")

        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device="cuda").manual_seed(int(seed_textbox))
        
        try:
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


def ui():
    controller = EasyAnimateController()

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
                    choices=["v1", "v2"],
                    value="v2",
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
            
            prompt_textbox = gr.Textbox(label="Prompt (正向提示词)", lines=2, value="This video shows the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene")
            negative_prompt_textbox = gr.Textbox(label="Negative prompt (负向提示词)", lines=2, value="The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion." )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method (采样器种类)", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps (生成步数)", value=50, minimum=10, maximum=100, step=1)
                        
                    width_slider     = gr.Slider(label="Width (视频宽度)",            value=672, minimum=128, maximum=1280, step=16)
                    height_slider    = gr.Slider(label="Height (视频高度)",           value=384, minimum=128, maximum=1280, step=16)
                    with gr.Row():
                        is_image      = gr.Checkbox(False, label="Generate Image (是否生成图片)")
                        length_slider = gr.Slider(label="Animation length (视频帧数)", value=144, minimum=9,   maximum=144,  step=9)
                    cfg_scale_slider = gr.Slider(label="CFG Scale (引导系数)",        value=7.0, minimum=0,   maximum=20)
                    
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

            is_image.change(
                lambda x: gr.update(visible=not x),
                inputs=[is_image],
                outputs=[length_slider],
            )
            easyanimate_edition_dropdown.change(
                fn=controller.update_edition, 
                inputs=[easyanimate_edition_dropdown], 
                outputs=[
                    easyanimate_edition_dropdown, 
                    diffusion_transformer_dropdown, 
                    motion_module_dropdown, 
                    motion_module_refresh_button, 
                    is_image, 
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
                    width_slider, 
                    height_slider, 
                    is_image, 
                    length_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller


class EasyAnimateController_Modelscope:
    def __init__(self, edition, config_path, model_name, savedir_sample):
        # Config and model path
        weight_dtype = torch.bfloat16
        self.savedir_sample = savedir_sample
        os.makedirs(self.savedir_sample, exist_ok=True)

        self.edition = edition
        self.inference_config = OmegaConf.load(config_path)
        # Get Transformer
        self.transformer = Transformer3DModel.from_pretrained_2d(
            model_name, 
            subfolder="transformer",
            transformer_additional_kwargs=OmegaConf.to_container(self.inference_config['transformer_additional_kwargs'])
        ).to(weight_dtype)
        if OmegaConf.to_container(self.inference_config['vae_kwargs'])['enable_magvit']:
            Choosen_AutoencoderKL = AutoencoderKLMagvit
        else:
            Choosen_AutoencoderKL = AutoencoderKL
        self.vae = Choosen_AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(weight_dtype)
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name, 
            subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_name, 
            subfolder="text_encoder", 
            torch_dtype=weight_dtype
        )
        self.pipeline = EasyAnimatePipeline(
            vae=self.vae, 
            text_encoder=self.text_encoder, 
            tokenizer=self.tokenizer, 
            transformer=self.transformer,
            scheduler=scheduler_dict["Euler"](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        )
        self.pipeline.enable_model_cpu_offload()
        print("Update diffusion transformer done")

    def generate(
        self,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        height_slider, 
        is_image, 
        length_slider, 
        cfg_scale_slider, 
        seed_textbox
    ):    
        if is_xformers_available(): self.transformer.enable_xformers_memory_efficient_attention()

        self.pipeline.scheduler = scheduler_dict[sampler_dropdown](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        self.pipeline.to("cuda")

        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device="cuda").manual_seed(int(seed_textbox))
        
        try:
            sample = self.pipeline(
                prompt_textbox,
                negative_prompt = negative_prompt_textbox,
                num_inference_steps = sample_step_slider,
                guidance_scale = cfg_scale_slider,
                width = width_slider,
                height = height_slider,
                video_length = length_slider if not is_image else 1,
                generator = generator
            ).videos
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            return gr.update(), gr.update(), f"Error. error information is {str(e)}"

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
            if gradio_version_is_above_4:
                return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
            else:
                return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            save_videos_grid(sample, save_sample_path, fps=12 if self.edition == "v1" else 24)
            if gradio_version_is_above_4:
                return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
            else:
                return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"


def ui_modelscope(edition, config_path, model_name, savedir_sample):
    controller = EasyAnimateController_Modelscope(edition, config_path, model_name, savedir_sample)

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
            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="This video shows the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene")
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion. " )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=50, minimum=10, maximum=100, step=1)
                    
                    if edition == "v1":
                        width_slider     = gr.Slider(label="Width",            value=512, minimum=384, maximum=704, step=32)
                        height_slider    = gr.Slider(label="Height",           value=512, minimum=384, maximum=704, step=32)
                        with gr.Row():
                            is_image      = gr.Checkbox(False, label="Generate Image", visible=False)
                        length_slider    = gr.Slider(label="Animation length", value=80,  minimum=40,  maximum=96,   step=1)
                        cfg_scale_slider = gr.Slider(label="CFG Scale",        value=6.0, minimum=0,   maximum=20)
                    else:
                        width_slider     = gr.Slider(label="Width",            value=672, minimum=256, maximum=704, step=16)
                        height_slider    = gr.Slider(label="Height",           value=384, minimum=256, maximum=704, step=16)
                        with gr.Column():
                            gr.Markdown(
                                """                    
                                To ensure the efficiency of the trial, we will limit the frame rate to no more than 81.
                                If you want to experience longer video generation, you can go to our [Github](https://github.com/aigc-apps/EasyAnimate/).
                                """
                            )
                            with gr.Row():
                                is_image      = gr.Checkbox(False, label="Generate Image")
                                length_slider = gr.Slider(label="Animation length", value=72, minimum=9,   maximum=81,  step=9)
                        cfg_scale_slider = gr.Slider(label="CFG Scale",        value=7.0, minimum=0,   maximum=20)
                    
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

            is_image.change(
                lambda x: gr.update(visible=not x),
                inputs=[is_image],
                outputs=[length_slider],
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    width_slider, 
                    height_slider, 
                    is_image, 
                    length_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller


def post_eas(
    prompt_textbox, negative_prompt_textbox, 
    sampler_dropdown, sample_step_slider, width_slider, height_slider,
    is_image, length_slider, cfg_scale_slider, seed_textbox,
):
    datas = {
        "base_model_path": "none",
        "motion_module_path": "none",
        "lora_model_path": "none", 
        "lora_alpha_slider": 0.55, 
        "prompt_textbox": prompt_textbox, 
        "negative_prompt_textbox": negative_prompt_textbox, 
        "sampler_dropdown": sampler_dropdown, 
        "sample_step_slider": sample_step_slider, 
        "width_slider": width_slider, 
        "height_slider": height_slider, 
        "is_image": is_image,
        "length_slider": length_slider,
        "cfg_scale_slider": cfg_scale_slider,
        "seed_textbox": seed_textbox,
    }
    # Token可以在公网地址调用信息中获取，详情请参见通用公网调用部分。
    session = requests.session()
    session.headers.update({"Authorization": os.environ.get("EAS_TOKEN")})

    response = session.post(url=f'{os.environ.get("EAS_URL")}/easyanimate/infer_forward', json=datas)
    outputs = response.json()
    return outputs


class EasyAnimateController_EAS:
    def __init__(self, edition, config_path, model_name, savedir_sample):
        self.savedir_sample = savedir_sample
        os.makedirs(self.savedir_sample, exist_ok=True)

    def generate(
        self,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        height_slider, 
        is_image, 
        length_slider, 
        cfg_scale_slider, 
        seed_textbox
    ):
        outputs = post_eas(
            prompt_textbox, negative_prompt_textbox, 
            sampler_dropdown, sample_step_slider, width_slider, height_slider,
            is_image, length_slider, cfg_scale_slider, seed_textbox
        )
        base64_encoding = outputs["base64_encoding"]
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
            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="This video shows the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene")
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion. " )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=30, minimum=10, maximum=100, step=1)
                    
                    if edition == "v1":
                        width_slider     = gr.Slider(label="Width",            value=512, minimum=384, maximum=704, step=32)
                        height_slider    = gr.Slider(label="Height",           value=512, minimum=384, maximum=704, step=32)
                        with gr.Row():
                            is_image      = gr.Checkbox(False, label="Generate Image", visible=False)
                        length_slider    = gr.Slider(label="Animation length", value=80,  minimum=40,  maximum=96,   step=1)
                        cfg_scale_slider = gr.Slider(label="CFG Scale",        value=6.0, minimum=0,   maximum=20)
                    else:
                        width_slider     = gr.Slider(label="Width",            value=672, minimum=256, maximum=704, step=16)
                        height_slider    = gr.Slider(label="Height",           value=384, minimum=256, maximum=704, step=16)
                        with gr.Column():
                            gr.Markdown(
                                """                    
                                To ensure the efficiency of the trial, we will limit the frame rate to no more than 81.
                                If you want to experience longer video generation, you can go to our [Github](https://github.com/aigc-apps/EasyAnimate/).
                                """
                            )
                            with gr.Row():
                                is_image      = gr.Checkbox(False, label="Generate Image")
                                length_slider = gr.Slider(label="Animation length", value=72, minimum=9,   maximum=81,  step=9)
                        cfg_scale_slider = gr.Slider(label="CFG Scale",        value=7.0, minimum=0,   maximum=20)
                    
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

            is_image.change(
                lambda x: gr.update(visible=not x),
                inputs=[is_image],
                outputs=[length_slider],
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    width_slider, 
                    height_slider, 
                    is_image, 
                    length_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller