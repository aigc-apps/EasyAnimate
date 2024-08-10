import os
from typing import List

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets.utils import download_url
from transformers import AutoModel, AutoProcessor

# All metrics.
__all__ = ["AestheticScore", "CLIPScore"]

_MODELS = {
    "CLIP_ViT-L/14": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/ViT-L-14.pt",
    "Aesthetics_V2": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/sac%2Blogos%2Bava1-l14-linearMSE.pth",
}
_MD5 = {
    "CLIP_ViT-L/14": "096db1af569b284eb76b3881534822d9",
    "Aesthetics_V2": "b1047fd767a00134b8fd6529bf19521a",
}


# if you changed the MLP architecture during training, change it also here:
class _MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScore:
    """Compute LAION Aesthetics Score V2 based on openai/clip. Note that the default
    inference dtype with GPUs is fp16 in openai/clip.

    Ref:
    1. https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py.
    2. https://github.com/openai/CLIP/issues/30.
    """

    def __init__(self, root: str = "~/.cache/clip", device: str = "cpu"):
        # The CLIP model is loaded in the evaluation mode.
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        filename = "ViT-L-14.pt"
        download_url(_MODELS["CLIP_ViT-L/14"], self.root, filename=filename, md5=_MD5["CLIP_ViT-L/14"])
        self.clip_model, self.preprocess = clip.load(os.path.join(self.root, filename), device=device)
        self.device = device
        self._load_mlp()

    def _load_mlp(self):
        filename = "sac+logos+ava1-l14-linearMSE.pth"
        download_url(_MODELS["Aesthetics_V2"], self.root, filename=filename, md5=_MD5["Aesthetics_V2"])
        state_dict = torch.load(os.path.join(self.root, filename))
        self.mlp = _MLP(768)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.mlp.eval()

    def __call__(self, images: List[Image.Image], texts=None) -> List[float]:
        with torch.no_grad():
            images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            image_embs = F.normalize(self.clip_model.encode_image(images))
            scores = self.mlp(image_embs.float())  # torch.float16 -> torch.float32, [N, 1]
        return scores.squeeze().tolist()
    
    def __repr__(self) -> str:
        return "aesthetic_score"


class CLIPScore:
    """Compute CLIP scores for image-text pairs based on huggingface/transformers."""

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16,
        device: str = "cpu",
    ):
        self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.torch_dtype = torch_dtype
        self.device = device

    def __call__(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        assert len(images) == len(texts)
        image_inputs = self.processor(images=images, return_tensors="pt")  # {"pixel_values": }
        if self.torch_dtype == torch.float16:
            image_inputs["pixel_values"] = image_inputs["pixel_values"].half()
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)  # {"inputs_id": }
        image_inputs, text_inputs = image_inputs.to(self.device), text_inputs.to(self.device)
        with torch.no_grad():
            image_embs = F.normalize(self.model.get_image_features(**image_inputs))
            text_embs = F.normalize(self.model.get_text_features(**text_inputs))
            scores = text_embs @ image_embs.T  # [N, N]

        return scores.diagonal().tolist()
    
    def __repr__(self) -> str:
        return "clip_score"


if __name__ == "__main__":
    aesthetic_score = AestheticScore(device="cuda")
    clip_score = CLIPScore(device="cuda")

    paths = ["demo/splash_cl2_midframe.jpg"] * 3
    texts = ["a joker", "a woman", "a man"]
    images = [Image.open(p).convert("RGB") for p in paths]

    print(aesthetic_score(images))
    print(clip_score(images, texts))