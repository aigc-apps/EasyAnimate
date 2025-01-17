import copy

import numpy as np
import numpy as numpy
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.utils.utils import save_videos_grid
from torch.utils.data.dataset import Dataset
from einops import rearrange

dit_path            = "models/Diffusion_Transformer/EasyAnimateV5-7b-zh-InP"
video_path          = "asset/00000003.mp4"
height              = 512
width               = 512
sample_n_frames     = 9
mini_batch_encoder  = 9
mini_batch_decoder  = 81
fps                 = 24

weight_dtype = torch.float32
model = AutoencoderKLMagvit.from_pretrained(dit_path, subfolder="vae")
model.to(weight_dtype)
model.cuda()

while 1:
    # Get Video
    video_reader = VideoReader(video_path, width=width, height=height)
    video_length = len(video_reader)
    batch_index = np.linspace(0, sample_n_frames, sample_n_frames, dtype=int)
    pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(3, 0, 1, 2).contiguous()
    pixel_values = (pixel_values.unsqueeze(0) / 127.5 - 1).to(model.device).to(model.dtype)
    save_videos_grid(pixel_values.float().cpu(), "./gt.mp4", rescale=True, fps=fps)

    # Get latents
    new_pixel_values = []
    for i in range(0, pixel_values.shape[2], mini_batch_encoder):
        with torch.no_grad():
            pixel_values_bs = pixel_values[:, :, i: i + mini_batch_encoder, :, :]
            pixel_values_bs = model.encode(pixel_values_bs)[0]
            pixel_values_bs = pixel_values_bs.sample()
            new_pixel_values.append(pixel_values_bs)
    latents = torch.cat(new_pixel_values, dim = 2)

    # Decoder
    video = []
    for i in range(0, latents.shape[2], mini_batch_decoder):
        with torch.no_grad():
            start_index = i
            end_index = i + mini_batch_decoder
            latents_bs = model.decode(latents[:, :, start_index:end_index, :, :])[0]
            video.append(latents_bs)

    # Smooth
    video = torch.cat(video, 2).clamp(-1, 1).to(model.device).to(model.dtype)

    # Save grid
    save_videos_grid(video.float().cpu(), "./pr.mp4", rescale=True, fps=fps)
    import pdb
    pdb.set_trace()