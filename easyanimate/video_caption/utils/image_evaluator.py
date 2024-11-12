import os
from typing import Union

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets.utils import download_url
from transformers import AutoModel, AutoProcessor

from .siglip_v2_5 import convert_v2_5_from_siglip

# All metrics.
__all__ = ["AestheticScore", "AestheticScoreSigLIP", "CLIPScore"]

_MODELS = {
    "CLIP_ViT-L/14": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/ViT-L-14.pt",
    "Aesthetics_V2": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/sac%2Blogos%2Bava1-l14-linearMSE.pth",
    "aesthetic_predictor_v2_5": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/aesthetic_predictor_v2_5.pth",
}
_MD5 = {
    "CLIP_ViT-L/14": "096db1af569b284eb76b3881534822d9",
    "Aesthetics_V2": "b1047fd767a00134b8fd6529bf19521a",
    "aesthetic_predictor_v2_5": "c46eb8c29f714c9231dc630b8226842a",
}


def get_list_depth(lst):
    if isinstance(lst, list):
        return 1 + max(get_list_depth(item) for item in lst)
    else:
        return 0


def reshape_images(images: Union[list[list[Image.Image]], list[Image.Image]]):
    # Check the input sanity.
    depth = get_list_depth(images)
    if depth == 1:  # batch image input
        if not isinstance(images[0], Image.Image):
            raise ValueError("The item in 1D images should be Image.Image.")
        num_sampled_frames = None
    elif depth == 2:  # batch video input
        if not isinstance(images[0][0], Image.Image):
            raise ValueError("The item in 2D images (videos) should be Image.Image.")
        num_sampled_frames = len(images[0])
        if not all(len(video_frames) == num_sampled_frames for video_frames in images):
            raise ValueError("All item in 2D images should be with the same length.")
        # [batch_size, num_sampled_frames, H, W, C] => [batch_size * num_sampled_frames, H, W, C].
        reshaped_images = []
        for video_frames in images:
            reshaped_images.extend([frame for frame in video_frames])
        images = reshaped_images
    else:
        raise ValueError("The input images should be in 1/2D list.")
    
    return images, num_sampled_frames


def reshape_scores(scores: list[float], num_sampled_frames: int) -> list[float]:
    if isinstance(scores, list):
        if num_sampled_frames is not None: # Batch video input
            batch_size = len(scores) // num_sampled_frames
            scores = [
                scores[i * num_sampled_frames:(i + 1) * num_sampled_frames]
                for i in range(batch_size)
            ]
        return scores
    else:
        return [scores]


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

    def __call__(self, images: Union[list[list[Image.Image]], list[Image.Image]], texts=None) -> list[float]:
        images, num_sampled_frames = reshape_images(images)

        with torch.no_grad():
            images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            image_embs = F.normalize(self.clip_model.encode_image(images))
            scores = self.mlp(image_embs.float())  # torch.float16 -> torch.float32, [N, 1]
        
        scores = scores.squeeze().tolist()  # scalar or list
        return reshape_scores(scores, num_sampled_frames)
    
    def __repr__(self) -> str:
        return "aesthetic_score"


class AestheticScoreSigLIP:
    """Compute Aesthetics Score V2.5 based on google/siglip-so400m-patch14-384.

    Ref:
    1. https://github.com/discus0434/aesthetic-predictor-v2-5.
    2. https://github.com/discus0434/aesthetic-predictor-v2-5/issues/2.
    """

    def __init__(
        self,
        root: str = "~/.cache/clip",
        device: str = "cpu",
        torch_dtype=torch.float16
    ):
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        filename = "aesthetic_predictor_v2_5.pth"
        download_url(_MODELS["aesthetic_predictor_v2_5"], self.root, filename=filename, md5=_MD5["aesthetic_predictor_v2_5"])
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            predictor_name_or_path=os.path.join(self.root, filename),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = self.model.to(device=device, dtype=torch_dtype)
        self.device = device
        self.torch_dtype = torch_dtype

    def __call__(self, images: Union[list[list[Image.Image]], list[Image.Image]], texts=None) -> list[float]:
        images, num_sampled_frames = reshape_images(images)

        pixel_values = self.preprocessor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, self.torch_dtype)
        with torch.no_grad():
            scores = self.model(pixel_values).logits.squeeze().float().cpu().numpy()
        
        scores = scores.squeeze().tolist()  # scalar or list
        return reshape_scores(scores, num_sampled_frames)
    
    def __repr__(self) -> str:
        return "aesthetic_score_siglip"


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

    def __call__(self, images: Union[list[list[Image.Image]], list[Image.Image]], texts: list[str]) -> list[float]:
        assert len(images) == len(texts)
        images, num_sampled_frames = reshape_images(images)
        # Expand texts in the batch video input case.
        if num_sampled_frames is not None:
            texts = [[text] * num_sampled_frames for text in texts]
            texts = [item for sublist in texts for item in sublist]

        image_inputs = self.processor(images=images, return_tensors="pt")  # {"pixel_values": }
        if self.torch_dtype == torch.float16:
            image_inputs["pixel_values"] = image_inputs["pixel_values"].half()
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)  # {"inputs_id": }
        image_inputs, text_inputs = image_inputs.to(self.device), text_inputs.to(self.device)
        with torch.no_grad():
            image_embs = F.normalize(self.model.get_image_features(**image_inputs))
            text_embs = F.normalize(self.model.get_text_features(**text_inputs))
            scores = text_embs @ image_embs.T  # [N, N]

        scores = scores.squeeze().tolist()  # scalar or list
        return reshape_scores(scores, num_sampled_frames)
    
    def __repr__(self) -> str:
        return "clip_score"


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from .video_dataset import VideoDataset, collate_fn

    aesthetic_score = AestheticScore(device="cuda")
    aesthetic_score_siglip = AestheticScoreSigLIP(device="cuda")
    # clip_score = CLIPScore(device="cuda")

    paths = ["your_image_path"] * 3
    # texts = ["a joker", "a woman", "a man"]
    images = [Image.open(p).convert("RGB") for p in paths]

    print(aesthetic_score(images))
    # print(clip_score(images, texts))

    test_dataset = VideoDataset(
        dataset_inputs={"video_path": ["your_video_path"] * 3},
        sample_method="mid",
        num_sampled_frames=2
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=collate_fn)

    for idx, batch in enumerate(tqdm(test_loader)):
        batch_frame = batch["sampled_frame"]
        print(aesthetic_score_siglip(batch_frame))