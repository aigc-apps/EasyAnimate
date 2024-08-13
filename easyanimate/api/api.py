import io
import gc
import base64
import torch
import gradio as gr
import tempfile
import hashlib
import os

from fastapi import FastAPI
from io import BytesIO
from PIL import Image

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

def save_base64_video(base64_string):
    video_data = base64.b64decode(base64_string)

    md5_hash = hashlib.md5(video_data).hexdigest()
    filename = f"{md5_hash}.mp4"  
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, 'wb') as video_file:
        video_file.write(video_data)

    return file_path

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
        resize_method = datas.get('resize_method', "Generate by")
        width_slider = datas.get('width_slider', 672)
        height_slider = datas.get('height_slider', 384)
        base_resolution = datas.get('base_resolution', 512)
        is_image = datas.get('is_image', False)
        generation_method = datas.get('generation_method', False)
        length_slider = datas.get('length_slider', 144)
        overlap_video_length = datas.get('overlap_video_length', 4)
        partial_video_length = datas.get('partial_video_length', 72)
        cfg_scale_slider = datas.get('cfg_scale_slider', 6)
        start_image = datas.get('start_image', None)
        end_image = datas.get('end_image', None)
        validation_video = datas.get('validation_video', None)
        denoise_strength = datas.get('denoise_strength', 0.70)
        seed_textbox = datas.get("seed_textbox", 43)

        generation_method = "Image Generation" if is_image else generation_method

        if start_image is not None:
            start_image = base64.b64decode(start_image)
            start_image = [Image.open(BytesIO(start_image))]
        
        if end_image is not None:
            end_image = base64.b64decode(end_image)
            end_image = [Image.open(BytesIO(end_image))]

        if validation_video is not None:
            validation_video = save_base64_video(validation_video)

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
                is_api = True,
            )
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            save_sample_path = ""
            comment = f"Error. error information is {str(e)}"
            return {"message": comment}
        
        if save_sample_path != "":
            return {"message": comment, "save_sample_path": save_sample_path, "base64_encoding": encode_file_to_base64(save_sample_path)}
        else:
            return {"message": comment, "save_sample_path": save_sample_path}