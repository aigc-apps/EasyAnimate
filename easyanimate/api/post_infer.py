import base64
import json
import sys
import time
from datetime import datetime
from io import BytesIO

import cv2
import requests
import base64


def post_diffusion_transformer(diffusion_transformer_path, url='http://127.0.0.1:7860'):
    datas = json.dumps({
        "diffusion_transformer_path": diffusion_transformer_path
    })
    r = requests.post(f'{url}/easyanimate/update_diffusion_transformer', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data

def post_update_edition(edition, url='http://0.0.0.0:7860'):
    datas = json.dumps({
        "edition": edition
    })
    r = requests.post(f'{url}/easyanimate/update_edition', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data

def post_infer(generation_method, length_slider, url='http://127.0.0.1:7860'):
    datas = json.dumps({
        "base_model_path": "none",
        "motion_module_path": "none",
        "lora_model_path": "none", 
        "lora_alpha_slider": 0.55, 
        "prompt_textbox": "This video shows Mount saint helens, washington - the stunning scenery of a rocky mountains during golden hours - wide shot. A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea.", 
        "negative_prompt_textbox": "Strange motion trajectory, a poor composition and deformed video, worst quality, normal quality, low quality, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera", 
        "sampler_dropdown": "Euler", 
        "sample_step_slider": 30, 
        "width_slider": 672, 
        "height_slider": 384, 
        "generation_method": "Video Generation",
        "length_slider": length_slider,
        "cfg_scale_slider": 6,
        "seed_textbox": 43,
    })
    r = requests.post(f'{url}/easyanimate/infer_forward', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data

if __name__ == '__main__':
    # initiate time
    now_date    = datetime.now()
    time_start  = time.time()  
    
    # -------------------------- #
    #  Step 1: update edition
    # -------------------------- #
    edition = "v3"
    outputs = post_update_edition(edition)
    print('Output update edition: ', outputs)

    # -------------------------- #
    #  Step 2: update edition
    # -------------------------- #
    diffusion_transformer_path = "models/Diffusion_Transformer/EasyAnimateV3-XL-2-512x512"
    outputs = post_diffusion_transformer(diffusion_transformer_path)
    print('Output update edition: ', outputs)

    # -------------------------- #
    #  Step 3: infer
    # -------------------------- #
    # "Video Generation" and "Image Generation"
    generation_method = "Video Generation"
    length_slider = 72
    outputs = post_infer(generation_method, length_slider)
    
    # Get decoded data
    outputs = json.loads(outputs)
    base64_encoding = outputs["base64_encoding"]
    decoded_data = base64.b64decode(base64_encoding)

    is_image = True if generation_method == "Image Generation" else False
    if is_image or length_slider == 1:
        file_path = "1.png"
    else:
        file_path = "1.mp4"
    with open(file_path, "wb") as file:
        file.write(decoded_data)
        
    # End of record time
    # The calculated time difference is the execution time of the program, expressed in seconds / s
    time_end = time.time()  
    time_sum = (time_end - time_start) % 60 
    print('# --------------------------------------------------------- #')
    print(f'#   Total expenditure: {time_sum}s')
    print('# --------------------------------------------------------- #')