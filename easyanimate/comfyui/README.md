# ComfyUI EasyAnimate
Easily use EasyAnimate inside ComfyUI!

[![Arxiv Page](https://img.shields.io/badge/Arxiv-Page-red)](https://arxiv.org/abs/2405.18991)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://easyanimate.github.io/)
[![Modelscope Studio](https://img.shields.io/badge/Modelscope-Studio-blue)](https://modelscope.cn/studios/PAI/EasyAnimate/summary)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/EasyAnimate)

- [Installation](#1-installation)
- [Node types](#node-types)
- [Example workflows](#example-workflows)
    - [Image to video](#image-to-video)
    - [Image to video generation (high FPS w/ frame interpolation)](#image-to-video-generation-high-fps-w-frame-interpolation)

## 1. Installation

### Option 1: Install via ComfyUI Manager
TBD

### Option 2: Install manually
```
cd ComfyUI/custom_nodes/
git clone https://github.com/aigc-apps/EasyAnimate.git
cd EasyAnimate/
python install.py
```

### 2. Download models into `ComfyUI/models/EasyAnimate/`
EasyAnimateV3:
| Name | Type | Storage Space | Url | Hugging Face | Description |
|--|--|--|--|--|--|
| EasyAnimateV3-XL-2-InP-512x512.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-512x512) | EasyAnimateV3 official weights for 512x512 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-768x768.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-768x768.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-768x768) | EasyAnimateV3 official weights for 768x768 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-960x960.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-960x960.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-960x960) | EasyAnimateV3 official weights for 960x960 text and  image to video resolution. Training with 144 frames and fps 24 |

## Node types
- **LoadEasyAnimateModel**
    - Loads the EasyAnimate model
- **TextBox**
    - Write the prompt for EasyAnimate model
- **EasyAnimateI2VSampler**
    - EasyAnimate Sampler for Image to Video 
- **EasyAnimateT2VSampler**
    - EasyAnimate Sampler for Text to Video

## Example workflows

### Image to video
Our ui is shown as follow:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/comfyui_i2v.jpg)

You can run the demo using following photo:
![demo image](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/firework.png)


### Image to video generation (high FPS w/ frame interpolation)
Our ui is shown as follow:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/comfyui_t2v.jpg)