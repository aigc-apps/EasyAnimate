import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.datasets.utils import download_url

from .longclip import longclip
from .viclip import get_viclip
from .video_utils import extract_frames

# All metrics.
__all__ = ["VideoCLIPXLScore"]

_MODELS = {
    "ViClip-InternVid-10M-FLT": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/ViClip-InternVid-10M-FLT.pth",
    "LongCLIP-L": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/longclip-L.pt",
    "VideoCLIP-XL-v2": "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/clip/VideoCLIP-XL-v2.bin",
}
_MD5 = {
    "ViClip-InternVid-10M-FLT": "b1ebf538225438b3b75e477da7735cd0",
    "LongCLIP-L": "5478b662f6f85ca0ebd4bb05f9b592f3",
    "VideoCLIP-XL-v2": "cebda0bab14b677ec061a57e80791f35",
}

def normalize(
    data: np.array,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225]
):
    v_mean = np.array(mean).reshape(1, 1, 3)
    v_std = np.array(std).reshape(1, 1, 3)

    return (data / 255.0 - v_mean) / v_std


class VideoCLIPXL(nn.Module):
    def __init__(self, root: str = "~/.cache/clip"):
        super(VideoCLIPXL, self).__init__()

        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        k = "LongCLIP-L"
        filename = os.path.basename(_MODELS[k])
        download_url(_MODELS[k], self.root, filename=filename, md5=_MD5[k])
        self.model = longclip.load(os.path.join(self.root, filename), device="cpu")[0].float()

        k = "ViClip-InternVid-10M-FLT"
        filename = os.path.basename(_MODELS[k])
        download_url(_MODELS[k], self.root, filename=filename, md5=_MD5[k])
        self.viclip_model = get_viclip("l", os.path.join(self.root, filename))["viclip"].float()

        # delete unused encoder
        del self.model.visual
        del self.viclip_model.text_encoder


class VideoCLIPXLScore():
    def __init__(self, root: str = "~/.cache/clip", device: str = "cpu"):
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        k = "VideoCLIP-XL-v2"
        filename = os.path.basename(_MODELS[k])
        download_url(_MODELS[k], self.root, filename=filename, md5=_MD5[k])
        self.model = VideoCLIPXL()
        state_dict = torch.load(os.path.join(self.root, filename), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        self.device = device
    
    def __call__(self, videos: List[List[Image.Image]], texts: List[str]):
        assert len(videos) == len(texts)

        # Use cv2.resize in accordance with the official demo. Resize and Normalize => B * [T, 224, 224, 3].
        videos = [[cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR) for f in v] for v in videos]
        resize_videos = [[cv2.resize(f, (224, 224)) for f in v] for v in videos]
        resize_normalizied_videos = [normalize(np.stack(v)) for v in resize_videos]

        video_inputs = torch.stack([torch.from_numpy(v) for v in resize_normalizied_videos])
        video_inputs = video_inputs.float().permute(0, 1, 4, 2, 3).to(self.device, non_blocking=True)  # BTCHW

        with torch.no_grad():
            vid_features = torch.stack(
                [self.model.viclip_model.get_vid_features(x.unsqueeze(0)).float() for x in video_inputs]
            )
            vid_features.squeeze_()
            # vid_features = self.model.viclip_model.get_vid_features(video_inputs).float()
            text_inputs = longclip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.model.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            scores = text_features @ vid_features.T
        
        return scores.tolist() if len(videos) == 1 else scores.diagonal().tolist()
    
    def __repr__(self):
        return "videoclipxl_score"


if __name__ == "__main__":
    videos = ["your_video_path"] * 3
    texts = [
        "a joker",
        "glasses and flower",
        "The video opens with a view of a white building with multiple windows, partially obscured by leafless tree branches. The scene transitions to a closer view of the same building, with the tree branches more prominent in the foreground. The focus then shifts to a street sign that reads 'Abesses' in bold, yellow letters against a green background. The sign is attached to a metal structure, possibly a tram or bus stop. The sign is illuminated by a light source above it, and the background reveals a glimpse of the building and tree branches from earlier shots. The colors are muted, with the yellow sign standing out against the grey and green hues."
    ]

    video_clip_xl_score = VideoCLIPXLScore(device="cuda")
    batch_frames = []
    for v in videos:
        sampled_frames = extract_frames(v, sample_method="uniform", num_sampled_frames=8)[1]
        batch_frames.append(sampled_frames)
    print(video_clip_xl_score(batch_frames, texts))