import os
import time
from datetime import datetime
from typing import List, Union
from pathlib import Path

import sglang as sgl
from PIL import Image

from utils.logger import logger

TMP_DIR = "./tmp"


def get_timestamp():
    timestamp_ns = int(time.time_ns())
    milliseconds = timestamp_ns // 1000000
    formatted_time = datetime.fromtimestamp(milliseconds / 1000).strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]

    return formatted_time


class LLaVASRT:
    def __init__(self, device: str = "cuda:0", quantized: bool = True):
        self.runtime = sgl.Runtime(model_path="liuhaotian/llava-v1.6-vicuna-7b", tokenizer_path="llava-hf/llava-1.5-7b-hf")
        sgl.set_default_backend(self.runtime)
        logger.info(
            f"Start the SGLang runtime for llava-v1.6-vicuna-7b with chat template: {self.runtime.endpoint.chat_template.name}. "
            "Input parameter device and quantized do not take effect."
        )
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR, exist_ok=True)

    @sgl.function
    def image_qa(s, prompt: str, image: str):
        s += sgl.user(sgl.image(image) + prompt)
        s += sgl.assistant(sgl.gen("answer"))

    def __call__(self, prompt: Union[str, List[str]], image: Union[str, Image.Image, List[str]]):
        pil_input_flag = False
        if isinstance(prompt, str) and (isinstance(image, str) or isinstance(image, Image.Image)):
            if isinstance(image, Image.Image):
                pil_input_flag = True
                image_path = os.path.join(TMP_DIR, get_timestamp() + ".jpg")
                image.save(image_path)
            state = self.image_qa.run(prompt=prompt, image=image, max_new_tokens=256)
            # Post-process.
            if pil_input_flag:
                os.remove(image)

            return state["answer"], state
        elif isinstance(prompt, list) and isinstance(image, list):
            assert len(prompt) == len(image)
            if isinstance(image[0], Image.Image):
                pil_input_flag = True
                image_path = [os.path.join(TMP_DIR, get_timestamp() + f"-{i}" + ".jpg") for i in range(len(image))]
                for i in range(len(image)):
                    image[i].save(image_path[i])
                image = image_path
            batch_query = [{"prompt": p, "image": img} for p, img in zip(prompt, image)]
            state = self.image_qa.run_batch(batch_query, max_new_tokens=256)
            # Post-process.
            if pil_input_flag:
                for i in range(len(image)):
                    os.remove(image[i])

            return [s["answer"] for s in state], state
        else:
            raise ValueError("Input prompt and image must be both strings or list of strings with the same length.")
    
    def __del__(self):
        self.runtime.shutdown()


if __name__ == "__main__":
    image_folder = "demo/"
    wildcard_list = ["*.jpg", "*.png"]
    image_list = []
    for wildcard in wildcard_list:
        image_list.extend([str(image_path) for image_path in Path(image_folder).glob(wildcard)])
    # SGLang need the exclusive GPU and cannot re-initialize CUDA in forked subprocess.
    llava_srt = LLaVASRT()
    # Batch inference.
    llava_srt_prompt = ["Please describe this image in detail."] * len(image_list)
    response, _ = llava_srt(llava_srt_prompt, image_list)
    print(response)
    llava_srt_prompt = "Please describe this image in detail."
    for image in image_list:
        response, _ = llava_srt(llava_srt_prompt, image)
        print(image, response)