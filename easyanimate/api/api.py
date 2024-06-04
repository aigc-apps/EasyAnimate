import io
import gc
import base64
import torch
import gradio as gr

from fastapi import FastAPI
from io import BytesIO

# Function to encode a file to Base64
def encode_file_to_base64(file_path):
    with open(file_path, "rb") as file:
        # Encode the data to Base64
        file_base64 = base64.b64encode(file.read())
        return file_base64

def update_edition_api(_: gr.Blocks, app: FastAPI, controller):
    @app.post("/easyanimate/update_edition")
    def _update_edition_api(
        datas: dict,
    ):
        edition = datas.get('edition', 'v2')

        try:
            controller.update_edition(
                edition
            )
            comment = "Success"
        except Exception as e:
            torch.cuda.empty_cache()
            comment = f"Error. error information is {str(e)}"

        return {"message": comment}

def update_diffusion_transformer_api(_: gr.Blocks, app: FastAPI, controller):
    @app.post("/easyanimate/update_diffusion_transformer")
    def _update_diffusion_transformer_api(
        datas: dict,
    ):
        diffusion_transformer_path = datas.get('diffusion_transformer_path', 'none')

        try:
            controller.update_diffusion_transformer(
                diffusion_transformer_path
            )
            comment = "Success"
        except Exception as e:
            torch.cuda.empty_cache()
            comment = f"Error. error information is {str(e)}"

        return {"message": comment}

def infer_forward_api(_: gr.Blocks, app: FastAPI, controller):
    @app.post("/easyanimate/infer_forward")
    def _infer_forward_api(
        datas: dict,
    ):
        base_model_path = datas.get('base_model_path', 'none')
        motion_module_path = datas.get('motion_module_path', 'none')
        lora_model_path = datas.get('lora_model_path', 'none')
        lora_alpha_slider = datas.get('lora_alpha_slider', 0.55)
        prompt_textbox = datas.get('prompt_textbox', None)
        negative_prompt_textbox = datas.get('negative_prompt_textbox', 'The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion.')
        sampler_dropdown = datas.get('sampler_dropdown', 'Euler')
        sample_step_slider = datas.get('sample_step_slider', 30)
        width_slider = datas.get('width_slider', 672)
        height_slider = datas.get('height_slider', 384)
        is_image = datas.get('is_image', False)
        length_slider = datas.get('length_slider', 144)
        cfg_scale_slider = datas.get('cfg_scale_slider', 6)
        seed_textbox = datas.get("seed_textbox", 43)

        try:
            save_sample_path, comment = controller.generate(
                "",
                base_model_path,
                motion_module_path,
                lora_model_path, 
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
                is_api = True,
            )
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            save_sample_path = ""
            comment = f"Error. error information is {str(e)}"
            return {"message": comment}

        return {"message": comment, "save_sample_path": save_sample_path, "base64_encoding": encode_file_to_base64(save_sample_path)}