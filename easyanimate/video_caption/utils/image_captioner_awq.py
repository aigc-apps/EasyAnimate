from pathlib import Path
from typing import Tuple

import auto_gptq
import torch
from auto_gptq.modeling import BaseGPTQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


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
