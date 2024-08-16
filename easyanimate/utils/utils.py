import os
import gc
import imageio
import numpy as np
import torch
import torchvision
import cv2
from einops import rearrange
from PIL import Image

def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)

def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 3:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -3:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -3:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 3:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size):
    if type(input_video_path) is str:
        cap = cv2.VideoCapture(input_video_path)
        input_video = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
            input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        input_video = input_video_path

    input_video = torch.from_numpy(np.array(input_video))[:video_length]
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

    input_video_mask = torch.zeros_like(input_video[:, :1])
    input_video_mask[:, :, :] = 255

    return  input_video, input_video_mask, None