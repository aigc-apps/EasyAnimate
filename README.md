# 📷 EasyAnimate | An End-to-End Solution for High-Resolution and Long Video Generation
😊 EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata. 

😊 Based on Sora like structure and DIT, we use transformer as a diffuser for video generation. We built easyanimate based on motion module, u-vit and slice-vae. In the future, we will try more training programs to improve the effect.

😊 Welcome!

[![Arxiv Page](https://img.shields.io/badge/Arxiv-Page-red)](https://arxiv.org/abs/2405.18991)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://easyanimate.github.io/)
[![Modelscope Studio](https://img.shields.io/badge/Modelscope-Studio-blue)](https://modelscope.cn/studios/PAI/EasyAnimate/summary)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/EasyAnimate)
[![Discord Page](https://img.shields.io/badge/Discord-Page-blue)](https://discord.gg/UzkpB4Bn)

English | [简体中文](./README_zh-CN.md)

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [How to use](#how-to-use)
- [Model zoo](#model-zoo)
- [Algorithm Detailed](#algorithm-detailed)
- [TODO List](#todo-list)
- [Contact Us](#contact-us)
- [Reference](#reference)
- [License](#license)

# Introduction
EasyAnimate is a pipeline based on the transformer architecture that can be used to generate AI photos and videos, train baseline models and Lora models for the Diffusion Transformer. We support making predictions directly from the pre-trained EasyAnimate model to generate videos of about different resolutions, 6 seconds with 24 fps (1 ~ 144 frames, in the future, we will support longer videos). Users are also supported to train their own baseline models and Lora models to perform certain style transformations. 

We will support quick pull-ups from different platforms, refer to [Quick Start](#quick-start).

What's New:
- Updated to version 4, supporting a maximum resolution of 1024x1024, 144 frames, 6 seconds, and 24fps video generation. It also supports larger resolutions of 1280x1280 with video generation at 96 frames. Supports the generation of videos from text, images, and videos. A single model can support arbitrary resolutions from 512 to 1280 and supports bilingual predictions in Chinese and English. [ 2024.08.15 ]
- Support ComfyUI, please refer to [ComfyUI README](comfyui/README.md) for details. [ 2024.07.12 ]
- Updated to v3, supports up to 720p 144 frames (960x960, 6s, 24fps) video generation, and supports text and image generated video models. [ 2024.07.01 ]
- ModelScope-Sora "Data Directors" creative sprint has been annouced using EasyAnimate as the training backbone to investigate the influence of data preprocessing. Please visit the competition's [official website](https://tianchi.aliyun.com/competition/entrance/532219) for more information. [ 2024.06.17 ]
- Updated to v2, supports a maximum of 144 frames (768x768, 6s, 24fps) for generation. [ 2024.05.26 ]
- Create Code! Support for Windows and Linux Now. [ 2024.04.12 ]

Function：
- [Data Preprocessing](#data-preprocess)
- [Train VAE](#vae-train)
- [Train DiT](#dit-train)
- [Video Generation](#video-gen)

These are our generated results [GALLERY](scripts/Result_Gallery.md) (Click the image below to see the video):


[![Watch the video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/i2v_result.jpg)](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/EasyAnimate-v3-DemoShow.mp4)


Our UI interface is as follows:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/ui_v3.jpg)

# Quick Start
### 1. Cloud usage: AliyunDSW/Docker
#### a. From AliyunDSW
DSW has free GPU time, which can be applied once by a user and is valid for 3 months after applying.

Aliyun provide free GPU time in [Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1), get it and use in Aliyun PAI-DSW to start EasyAnimate within 5min!

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/easyanimate)

#### b. From ComfyUI
Our ComfyUI is as follows, please refer to [ComfyUI README](comfyui/README.md) for details.
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/comfyui_i2v.jpg)

#### c. From docker
If you are using docker, please make sure that the graphics card driver and CUDA environment have been installed correctly in your machine.

Then execute the following commands in this way:

EasyAnimateV4:
```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# clone code
git clone https://github.com/aigc-apps/EasyAnimate.git

# enter EasyAnimate's dir
cd EasyAnimate

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV4-XL-2-InP.tar.gz -O models/Diffusion_Transformer/EasyAnimateV4-XL-2-InP.tar.gz

cd models/Diffusion_Transformer/
tar -zxvf EasyAnimateV4-XL-2-InP.tar.gz
cd ../../
```

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# clone code
git clone https://github.com/aigc-apps/EasyAnimate.git

# enter EasyAnimate's dir
cd EasyAnimate

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar -O models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar

cd models/Diffusion_Transformer/
tar -xvf EasyAnimateV3-XL-2-InP-512x512.tar
cd ../../
```
</details>

<details>
  <summary>(Obsolete) EasyAnimateV2:</summary>

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# clone code
git clone https://github.com/aigc-apps/EasyAnimate.git

# enter EasyAnimate's dir
cd EasyAnimate

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV2-XL-2-512x512.tar -O models/Diffusion_Transformer/EasyAnimateV2-XL-2-512x512.tar

cd models/Diffusion_Transformer/
tar -xvf EasyAnimateV2-XL-2-512x512.tar
cd ../../
```
</details>

<details>
  <summary>(Obsolete) EasyAnimateV1:</summary>
  
```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# clone code
git clone https://github.com/aigc-apps/EasyAnimate.git

# enter EasyAnimate's dir
cd EasyAnimate

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar -O models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-768x768.tar -O models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-768x768.tar
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-960x960.tar -O models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-960x960.tar

cd models/Diffusion_Transformer/
tar -xvf EasyAnimateV3-XL-2-InP-512x512.tar
tar -xvf EasyAnimateV3-XL-2-InP-768x768.tar
tar -xvf EasyAnimateV3-XL-2-InP-960x960.tar
cd ../../
```
</details>

### 2. Local install: Environment Check/Downloading/Installation
#### a. Environment Check
We have verified EasyAnimate execution on the following environment:

The detailed of Windows:
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G

The detailed of Linux:
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

We need about 60GB available on disk (for saving weights), please check!

The video sizes of EasyAnimateV4 that can be generated by different graphics memory include:
| GPU memory | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 768x1344x72 | 768x1344x144 | 960x1680x96 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| 12GB | ⭕️ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ | ❌ |
| 16GB | ✅ | ✅ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ |
| 24GB | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

✅ indicates it can run with low_gpu_memory_mode=False, ⭕️ indicates it can run with low_gpu_memory_mode=True, ❌ indicates it cannot run. Note that running with low_gpu_memory_mode=True will be slower.

Some graphics cards, like the 2080ti and V100, do not support torch.bfloat16. If you are using one of these cards, you will need to modify the weight_dtype to torch.float16 in both the app.py and predict files to run the program.

The generation time of different GPUs at 25 steps is as follows:
| GPU | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 768x1344x72 | 768x1344x144 | 960x1680x96 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| A10 24GB | ~180s | ~370s | ~480s | ~1800s(⭕️) | ~1000s | ❌ | ❌ |
| A100 80GB | ~60s | ~180s | ~200s | ~600s | ~500s | ~1800s | ~1800s |

(⭕️) indicates that it can run with low_gpu_memory_mode=True, but at a slower speed, while ❌ indicates that it cannot run.

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>

The video sizes of EasyAnimateV3 that can be generated by different graphics memory include:
| GPU memory | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 720x1280x72 | 720x1280x144 |
|----------|----------|----------|----------|----------|----------|----------|
| 12GB | ⭕️ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ |
| 16GB | ✅ | ✅ | ⭕️ | ⭕️ | ⭕️ | ❌ |
| 24GB | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
</details>

#### b. Weights
We'd better place the [weights](#model-zoo) along the specified path:

EasyAnimateV4:
```
📦 models/
├── 📂 Diffusion_Transformer/
│   └── 📂 EasyAnimateV4-XL-2-InP/
├── 📂 Personalized_Model/
│   └── your trained trainformer model / your trained lora model (for UI load)
```

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>:

```
📦 models/
├── 📂 Diffusion_Transformer/
│   └── 📂 EasyAnimateV3-XL-2-InP-512x512/
├── 📂 Personalized_Model/
│   └── your trained trainformer model / your trained lora model (for UI load)
```
</details>

<details>
  <summary>(Obsolete) EasyAnimateV2:</summary>

```
📦 models/
├── 📂 Diffusion_Transformer/
│   └── 📂 EasyAnimateV2-XL-2-512x512/
├── 📂 Personalized_Model/
│   └── your trained trainformer model / your trained lora model (for UI load)
```
</details>

<details>
  <summary>(Obsolete) EasyAnimateV1:</summary>

  ```
  📦 models/
  ├── 📂 Diffusion_Transformer/
  │   └── 📂 PixArt-XL-2-512x512/
  ├── 📂 Motion_Module/
  │   └── 📄 easyanimate_v1_mm.safetensors
  ├── 📂 Personalized_Model/
  │   ├── 📄 easyanimate_portrait.safetensors
  │   └── 📄 easyanimate_portrait_lora.safetensors
  ```
</details>

# How to use

<h3 id="video-gen">1. Inference </h3>

#### a. Using Python Code
- Step 1: Download the corresponding [weights](#model-zoo) and place them in the models folder.
- Step 2: Modify prompt, neg_prompt, guidance_scale, and seed in the predict_t2v.py file.
- Step 3: Run the predict_t2v.py file, wait for the generated results, and save the results in the samples/easyanimate-videos folder.
- Step 4: If you want to combine other backbones you have trained with Lora, modify the predict_t2v.py and Lora_path in predict_t2v.py depending on the situation.

#### b. Using webui
- Step 1: Download the corresponding [weights](#model-zoo) and place them in the models folder.
- Step 2: Run the app.py file to enter the graph page.
- Step 3: Select the generated model based on the page, fill in prompt, neg_prompt, guidance_scale, and seed, click on generate, wait for the generated result, and save the result in the samples folder.

#### c. From ComfyUI
Please refer to [ComfyUI README](comfyui/README.md) for details.

### 2. Model Training
A complete EasyAnimate training pipeline should include data preprocessing, Video VAE training, and Video DiT training. Among these, Video VAE training is optional because we have already provided a pre-trained Video VAE.

<h4 id="data-preprocess">a. data preprocessing</h4>

We have provided a simple demo of training the Lora model through image data, which can be found in the [wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-Lora) for details.

A complete data preprocessing link for long video segmentation, cleaning, and description can refer to [README](./easyanimate/video_caption/README.md) in the video captions section. 

If you want to train a text to image and video generation model. You need to arrange the dataset in this format.

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

The json_of_internal_datasets.json is a standard JSON file. The file_path in the json can to be set as relative path, as shown in below:
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

You can also set the path as absolute path as follow:
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

<h4 id="vae-train">b. Video VAE training (optional)</h4>

Video VAE training is an optional option as we have already provided pre trained Video VAEs.
If you want to train video vae, you can refer to [README](easyanimate/vae/README.md) in the video vae section.

<h4 id="dit-train">c. Video DiT training </h4>
 
If the data format is relative path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

If the data format is absolute path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

Then, we run scripts/train.sh.
```sh
sh scripts/train.sh
```

For details on setting some parameters, please refer to [Readme Train](scripts/README_TRAIN.md) and [Readme Lora](scripts/README_TRAIN_LORA.md). 

<details>
  <summary>(Obsolete) EasyAnimateV1:</summary>
  If you want to train EasyAnimateV1. Please switch to the git branch v1.
</details>


# Model zoo
EasyAnimateV4:

We attempted to implement EasyAnimate using 3D full attention, but this structure performed moderately on slice VAE and incurred considerable training costs. As a result, the performance of version V4 did not significantly surpass that of version V3. Due to limited resources, we are migrating EasyAnimate to a retrained 16-channel MagVit to pursue better model performance.

| Name | Type | Storage Space | Url | Hugging Face | Description |
|--|--|--|--|--|--|
| EasyAnimateV4-XL-2-InP.tar.gz | EasyAnimateV4 | Before extraction: 8.9 GB \/ After extraction: 14.0 GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV4-XL-2-InP.tar.gz) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV4-XL-2-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 144 frames at a rate of 24 frames per second. |

EasyAnimateV3:

| Name | Type | Storage Space | Url | Hugging Face | Description |
|--|--|--|--|--|--|
| EasyAnimateV3-XL-2-InP-512x512.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-512x512) | EasyAnimateV3 official weights for 512x512 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-768x768.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-768x768.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-768x768) | EasyAnimateV3 official weights for 768x768 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-960x960.tar | EasyAnimateV3 | 18.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-960x960.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-960x960) | EasyAnimateV3 official weights for 960x960 text and  image to video resolution. Training with 144 frames and fps 24 |

<details>
  <summary>(Obsolete) EasyAnimateV2:</summary>
| Name | Type | Storage Space | Url | Hugging Face | Description |
|--|--|--|--|--|--|
| EasyAnimateV2-XL-2-512x512.tar | EasyAnimateV2 | 16.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV2-XL-2-512x512.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-512x512) | EasyAnimateV2 official weights for 512x512 resolution. Training with 144 frames and fps 24 |
| EasyAnimateV2-XL-2-768x768.tar | EasyAnimateV2 | 16.2GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV2-XL-2-768x768.tar) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-768x768) | EasyAnimateV2 official weights for 768x768 resolution. Training with 144 frames and fps 24 |
| easyanimatev2_minimalism_lora.safetensors | Lora of Pixart | 485.1MB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimatev2_minimalism_lora.safetensors) | - | A lora training with a specifial type images. Images can be downloaded from [Url](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v2/Minimalism.zip). |
</details>

<details>
  <summary>(Obsolete) EasyAnimateV1:</summary>

### 1、Motion Weights
| Name | Type | Storage Space | Url | Description |
|--|--|--|--|--| 
| easyanimate_v1_mm.safetensors | Motion Module | 4.1GB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Motion_Module/easyanimate_v1_mm.safetensors) | Training with 80 frames and fps 12 |

### 2、Other Weights
| Name | Type | Storage Space | Url | Description |
|--|--|--|--|--| 
| PixArt-XL-2-512x512.tar | Pixart | 11.4GB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/PixArt-XL-2-512x512.tar)| Pixart-Alpha official weights |
| easyanimate_portrait.safetensors | Checkpoint of Pixart | 2.3GB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait.safetensors) | Training with internal portrait datasets |
| easyanimate_portrait_lora.safetensors | Lora of Pixart | 654.0MB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait_lora.safetensors)| Training with internal portrait datasets |
</details>



# Algorithm Detailed
### 1. Data Preprocessing
**Video Cut**

For long video cut, EasyAnimate utilizes PySceneDetect to identify scene changes within the video and performs scene cutting based on certain threshold values to ensure consistency in the themes of the video segments. After cutting, we only keep segments with lengths ranging from 3 to 10 seconds for model training.

**Video Cleaning and Description**

Following SVD's data preparation process, EasyAnimate provides a simple yet effective data processing pipeline for high-quality data filtering and labeling. It also supports distributed processing to accelerate the speed of data preprocessing. The overall process is as follows:

- Duration filtering: Analyze the basic information of the video to filter out low-quality videos that are short in duration or low in resolution.
- Aesthetic filtering: Filter out videos with poor content (blurry, dim, etc.) by calculating the average aesthetic score of uniformly distributed 4 frames.
- Text filtering: Use easyocr to calculate the text proportion of middle frames to filter out videos with a large proportion of text.
- Motion filtering: Calculate interframe optical flow differences to filter out videos that move too slowly or too quickly.
- Text description: Recaption video frames using videochat2 and vila. PAI is also developing a higher quality video recaption model, which will be released for use as soon as possible.

### 2. Model Architecture
EasyAnimateV4:

We used [Hunyuan-DiT](https://github.com/Tencent/HunyuanDiT) as the underlying framework, and modified the VAE and DiT model structures on this basis to better support video generation. Please refer to the original resource page and follow the corresponding license.

The overall structure of EasyAnimateV4 is as follows:

EasyAnimateV4 includes two text encoders, Video VAE (video encoder and decoder), and Diffusion Transformer (DiT). The MT5 Encoder and multi-modal CLIP are used as text encoders. EasyAnimateV4 employs 3D global attention for video reconstruction, eliminating the separation of motion modules and base models as seen in V3. This ensures coherent frame generation and seamless motion transitions through global attention.

The pipeline structure of EasyAnimateV4 is as follows:

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/framework_v4.jpg" alt="ui" style="zoom:50%;" />

The foundational model structure of EasyAnimateV4 is as follows:

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/pipeline_v4.jpg" alt="ui" style="zoom:50%;" />

The Slice VAE exhibits some stuttering during scene changes because the later latents cannot fully access information from the preceding blocks during decoding. 

Referring to MagVit, we stored the results after convolution of the previous blocks. Except for the initial video block, each subsequent video block during convolution only accessed the features of the preceding video blocks, not the following ones. After this modification, the decoder's reconstruction results are smoother compared to the original Slice VAE.

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>
We have adopted [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) as the base model and modified the VAE and DiT model structures on this basis to better support video generation. The overall structure of EasyAnimate is as follows:

The diagram below outlines the pipeline of EasyAnimate. It includes the Text Encoder, Video VAE (video encoder and decoder), and Diffusion Transformer (DiT). The T5 Encoder is used as the text encoder. Other components are detailed in the sections below.

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/pipeline_v3.jpg" alt="ui" style="zoom:50%;" />

We have expanded the DiT framework originally designed for 2D image synthesis to accommodate the complexities of 3D video generation by incorporating a special motion module block named Hybrid Motion Module. 

In the motion module, we employ a combination of temporal attention and global attention to ensure the generation of coherent frames and seamless motion transitions.

Additionally, referencing U-ViT, it introduces a skip connection structure into EasyAnimate to further optimize deeper features by incorporating shallow features. A fully connected layer is also zero-initialized for each skip connection structure, allowing it to be applied as a plug-in module to previously trained and well-performing DiTs.

Moreover, it proposes Slice VAE, which addresses the memory difficulties encountered by MagViT when dealing with long and large videos, while also achieving greater compression in the temporal dimension during video encoding and decoding stages compared to MagViT.

For more details, please refer to [arxiv](https://arxiv.org/abs/2405.18991).
</details>

# TODO List
- Support model with larger params.

# Contact Us
1. Use Dingding to search group 77450006752 or Scan to join
2. You need to scan the image to join the WeChat group or if it is expired, add this student as a friend first to invite you.

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/dd.png" alt="ding group" width="30%"/>
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/wechat.jpg" alt="Wechat group" width="30%"/>
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/person.jpg" alt="Person" width="30%"/>


# Reference
- magvit: https://github.com/google-research/magvit
- PixArt: https://github.com/PixArt-alpha/PixArt-alpha
- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- Animatediff: https://github.com/guoyww/AnimateDiff
- ComfyUI-EasyAnimateWrapper: https://github.com/kijai/ComfyUI-EasyAnimateWrapper
- HunYuan DiT: https://github.com/tencent/HunyuanDiT

# License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).