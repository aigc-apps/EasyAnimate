import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import auto_gptq
import sglang as sgl
import torch
from auto_gptq.modeling import BaseGPTQForCausalLM
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import logger

TMP_DIR = "./tmp"


def get_timestamp():
    timestamp_ns = int(time.time_ns())
    milliseconds = timestamp_ns // 1000000
    formatted_time = datetime.fromtimestamp(milliseconds / 1000).strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]

    return formatted_time


class QwenVLChat:
    def __init__(self, device: str = "cuda:0", quantized: bool = False) -> None:
        if quantized:
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat-Int4", device_map=device, trust_remote_code=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True, fp16=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    def __call__(self, prompt: str, image: str) -> Tuple[str, str]:
        query = self.tokenizer.from_list_format([{"image": image}, {"text": prompt}])
        response, history = self.model.chat(self.tokenizer, query=query, history=[])

        return response, history


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        "vit",
        "vision_proj",
        "model.tok_embeddings",
        "model.norm",
        "output",
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]


class InternLMXComposer2:
    def __init__(self, device: str = "cuda:0", quantized: bool = True):
        if quantized:
            auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
            self.model = InternLMXComposer2QForCausalLM.from_quantized(
                "internlm/internlm-xcomposer2-vl-7b-4bit", trust_remote_code=True, device=device
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-xcomposer2-vl-7b-4bit", trust_remote_code=True)
        else:
            # Setting fp16=True does not work. See https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/discussions/1.
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    "internlm/internlm-xcomposer2-vl-7b", device_map=device, trust_remote_code=True
                )
                .eval()
                .to(torch.float16)
            )
            self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-xcomposer2-vl-7b", trust_remote_code=True)

    def __call__(self, prompt: str, image: str):
        if not prompt.startswith("<ImageHere>"):
            prompt = "<ImageHere>" + prompt
        with torch.cuda.amp.autocast(), torch.no_grad():
            response, history = self.model.chat(self.tokenizer, query=prompt, image=image, history=[], do_sample=False)
        return response, history


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
    qwen_vl_chat = QwenVLChat(device="cuda:0", quantized=True)
    qwen_vl_prompt = "Please describe this image in detail."
    for image in image_list:
        response, _ = qwen_vl_chat(qwen_vl_prompt, image)
        print(image, response)

    internlm2_vl = InternLMXComposer2(device="cuda:0", quantized=False)
    internlm2_vl_prompt = "Please describe this image in detail."
    for image in image_list:
        response, _ = internlm2_vl(internlm2_vl_prompt, image)
        print(image, response)

    # # SGLang need the exclusive GPU and cannot re-initialize CUDA in forked subprocess.
    # llava_srt = LLaVASRT()
    # # Batch inference.
    # llava_srt_prompt = ["Please describe this image in detail."] * len(image_list)
    # response, _ = llava_srt(llava_srt_prompt, image_list)
    # print(response)
    # Single inference.
    # llava_srt_prompt = "Please describe this image in detail."
    # for image in image_list:
    #     response, _ = llava_srt(llava_srt_prompt, image)
    #     print(image, response)
