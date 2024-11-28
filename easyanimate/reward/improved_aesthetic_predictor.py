import os

import torch
import torch.nn as nn
from transformers import CLIPModel
from torchvision.datasets.utils import download_url

URL = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Third_Party/sac%2Blogos%2Bava1-l14-linearMSE.pth"
FILENAME = "sac+logos+ava1-l14-linearMSE.pth"
MD5 = "b1047fd767a00134b8fd6529bf19521a"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)


class ImprovedAestheticPredictor(nn.Module):
    def __init__(self, encoder_path="openai/clip-vit-large-patch14", predictor_path=None):
        super().__init__()
        self.encoder = CLIPModel.from_pretrained(encoder_path)
        self.predictor = MLP()
        if predictor_path is None or not os.path.exists(predictor_path):
            download_url(URL, torch.hub.get_dir(), FILENAME, md5=MD5)
            predictor_path = os.path.join(torch.hub.get_dir(), FILENAME)
        state_dict = torch.load(predictor_path, map_location="cpu")
        self.predictor.load_state_dict(state_dict)
        self.eval()
    

    def forward(self, pixel_values):
        embed = self.encoder.get_image_features(pixel_values=pixel_values)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        return self.predictor(embed).squeeze(1)
