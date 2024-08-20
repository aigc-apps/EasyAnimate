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
The EasyAnimate repository needs to be placed at `ComfyUI/custom_nodes/EasyAnimate/`.

```
cd ComfyUI/custom_nodes/

# Git clone the easyanimate itself
git clone https://github.com/aigc-apps/EasyAnimate.git

# Git clone the video outout node
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

cd EasyAnimate/
python install.py
```

### 2. Download models into `ComfyUI/models/EasyAnimate/`

| Name | Type | Storage Space | Url | Hugging Face | Description |
| EasyAnimateV4-XL-2-InP.tar | EasyAnimateV4 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV4-XL-2-InP.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV4-XL-2-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 144 frames at a rate of 24 frames per second. |

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>

| Name | Type | Storage Space | Url | Hugging Face | Description |
|--|--|--|--|--|--|
| EasyAnimateV3-XL-2-InP-512x512.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-512x512) | EasyAnimateV3 official weights for 512x512 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-768x768.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-768x768.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-768x768) | EasyAnimateV3 official weights for 768x768 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-960x960.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-960x960.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-960x960) | EasyAnimateV3 official weights for 960x960 text and  image to video resolution. Training with 144 frames and fps 24 |
</details>

## Node types
- **LoadEasyAnimateModel**
    - Loads the EasyAnimate model
- **TextBox**
    - Write the prompt for EasyAnimate model
- **EasyAnimateI2VSampler**
    - EasyAnimate Sampler for Image to Video 
- **EasyAnimateT2VSampler**
    - EasyAnimate Sampler for Text to Video
- **EasyAnimateV2VSampler**
    - EasyAnimate Sampler for Video to Video

## Example workflows

### Video to video generation
Our ui is shown as follow, this is the [download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/easyanimatev4_workflow_v2v.json) of the json:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/comfyui_v2v.jpg)

You can run the demo using following video:
[demo video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/play_guitar.mp4)

### Image to video generation
Our ui is shown as follow, this is the [download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/easyanimatev4_workflow_i2v.json) of the json:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/comfyui_i2v.jpg)

You can run the demo using following photo:
![demo image](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/firework.png)

### Text to video generation
Our ui is shown as follow, this is the [download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/easyanimatev4_workflow_t2v.json) of the json:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/comfyui_t2v.jpg)

### Text to video generation With Lora
We have provided a v4 version of the portrait Lora for testing, and the specific download link is [Lora download Link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimatev4_minimalism_lora.safetensors).

You can put this Lora at ```ComfyUI/models/loras/easyanimate```.

Our ui is shown as follow, this is the [download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/easyanimatev4_workflow_lora.json) of the json:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v4/comfyui_lora.jpg)