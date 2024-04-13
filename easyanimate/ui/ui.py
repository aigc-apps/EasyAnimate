"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/app.py
"""
import gc
import json
import os
import random
from datetime import datetime
from glob import glob

import gradio as gr
import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import T5EncoderModel, T5Tokenizer

from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import save_videos_grid

sample_idx = 0
scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}

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
        self.diffusion_transformer_dir  = os.path.join(self.basedir, "models", "Diffusion_Transformer")
        self.motion_module_dir          = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        self.savedir                    = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample             = os.path.join(self.savedir, "sample")
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
        self.lora_model_path       = "none"
        
        self.weight_dtype = torch.float16
        self.inference_config = OmegaConf.load("config/easyanimate_video_motion_module_v1.yaml")

    def refresh_diffusion_transformer(self):
        self.diffusion_transformer_list = glob(os.path.join(self.diffusion_transformer_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.safetensors"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]

    def update_diffusion_transformer(self, diffusion_transformer_dropdown):
        print("Update diffusion transformer")
        if diffusion_transformer_dropdown == "none":
            return gr.Dropdown.update()
        self.vae = AutoencoderKL.from_pretrained(diffusion_transformer_dropdown, subfolder="vae", torch_dtype = self.weight_dtype)
        self.transformer = Transformer3DModel.from_pretrained_2d(
            diffusion_transformer_dropdown, 
            subfolder="transformer", 
            transformer_additional_kwargs=OmegaConf.to_container(self.inference_config.transformer_additional_kwargs)
        ).to(self.weight_dtype)
        self.tokenizer = T5Tokenizer.from_pretrained(diffusion_transformer_dropdown, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(diffusion_transformer_dropdown, subfolder="text_encoder", torch_dtype = self.weight_dtype)
        print("Update diffusion transformer done")
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        print("Update motion module")
        if motion_module_dropdown == "none":
            return gr.Dropdown.update()
        if self.transformer is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
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
            assert len(unexpected) == 0
            print("Update motion module done")
            return gr.Dropdown.update()

    def update_base_model(self, base_model_dropdown):
        print("Update base model")
        if base_model_dropdown == "none":
            return gr.Dropdown.update()
        if self.transformer is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)
            self.transformer.load_state_dict(base_model_state_dict, strict=False)
            print("Update base done")
            return gr.Dropdown.update()

    def update_lora_model(self, lora_model_dropdown):
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_path = lora_model_dropdown
        return gr.Dropdown.update()

    def generate(
        self,
        diffusion_transformer_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        length_slider, 
        height_slider, 
        cfg_scale_slider, 
        seed_textbox
    ):    
        global sample_idx
        if self.transformer is None:
            raise gr.Error(f"Please select a pretrained model path.")
        if motion_module_dropdown == "": 
            raise gr.Error(f"Please select a motion module.")

        if is_xformers_available(): self.transformer.enable_xformers_memory_efficient_attention()

        pipeline = EasyAnimatePipeline(
            vae=self.vae, 
            text_encoder=self.text_encoder, 
            tokenizer=self.tokenizer, 
            transformer=self.transformer,
            scheduler=scheduler_dict[sampler_dropdown](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        )
        if self.lora_model_path != "none":
            # lora part
            pipeline = merge_lora(pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            
        pipeline.to("cuda")

        if seed_textbox != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: torch.seed()
        seed = torch.initial_seed()
        
        try:
            sample = pipeline(
                prompt_textbox,
                negative_prompt     = negative_prompt_textbox,
                num_inference_steps = sample_step_slider,
                guidance_scale      = cfg_scale_slider,
                width               = width_slider,
                height              = height_slider,
                video_length        = length_slider,
            ).videos
        except Exception as e:
            # lora part
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if self.lora_model_path != "none":
                pipeline = unmerge_lora(pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            return gr.Video.update()

        # lora part
        if self.lora_model_path != "none":
            pipeline = unmerge_lora(pipeline, self.lora_model_path, multiplier=lora_alpha_slider)

        save_sample_path = os.path.join(self.savedir_sample, f"{sample_idx}.mp4")
        sample_idx += 1
        save_videos_grid(sample, save_sample_path, fps=12)
    
        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed": seed
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        return gr.Video.update(value=save_sample_path)


def ui():
    controller = EasyAnimateController()

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # EasyAnimate: Generate your animation easily
            [Github](https://github.com/aigc-apps/EasyAnimate/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model checkpoints (select pretrained model path first).
                """
            )
            with gr.Row():
                diffusion_transformer_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
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
                    return gr.Dropdown.update(choices=controller.diffusion_transformer_list)
                diffusion_transformer_refresh_button.click(fn=refresh_diffusion_transformer, inputs=[], outputs=[diffusion_transformer_dropdown])

            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module",
                    choices=controller.motion_module_list,
                    value="none",
                    interactive=True,
                )
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown], outputs=[motion_module_dropdown])
                
                motion_module_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_motion_module():
                    controller.refresh_motion_module()
                    return gr.Dropdown.update(choices=controller.motion_module_list)
                motion_module_refresh_button.click(fn=update_motion_module, inputs=[], outputs=[motion_module_dropdown])
                
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (optional)",
                    choices=controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )
                base_model_dropdown.change(fn=controller.update_base_model, inputs=[base_model_dropdown], outputs=[base_model_dropdown])
                
                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (optional)",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )
                lora_model_dropdown.change(fn=controller.update_lora_model, inputs=[lora_model_dropdown], outputs=[lora_model_dropdown])
                
                lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.55, minimum=0, maximum=2, interactive=True)
                
                personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.Dropdown.update(choices=controller.personalized_model_list),
                        gr.Dropdown.update(choices=["none"] + controller.personalized_model_list)
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[base_model_dropdown, lora_model_dropdown])

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for Generation.
                """
            )
            
            prompt_textbox = gr.Textbox(label="Prompt", lines=2)
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="Strange motion trajectory, a poor composition and deformed video, worst quality, normal quality, low quality, low resolution, duplicate and ugly" )
                
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=30, minimum=10, maximum=100, step=1)
                        
                    width_slider     = gr.Slider(label="Width",            value=512, minimum=256, maximum=1024, step=64)
                    height_slider    = gr.Slider(label="Height",           value=512, minimum=256, maximum=1024, step=64)
                    length_slider    = gr.Slider(label="Animation length", value=80,  minimum=16,   maximum=96,   step=1)
                    cfg_scale_slider = gr.Slider(label="CFG Scale",        value=6.0, minimum=0,   maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])

                    generate_button = gr.Button(value="Generate", variant='primary')
                    
                result_video = gr.Video(label="Generated Animation", interactive=False)

            generate_button.click(
                fn=controller.generate,
                inputs=[
                    diffusion_transformer_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_alpha_slider,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    width_slider, 
                    length_slider, 
                    height_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                ],
                outputs=[result_video]
            )
    return demo
