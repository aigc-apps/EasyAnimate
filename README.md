# 📷 EasyAnimate | An End-to-End Solution for High-Resolution and Long Video Generation
😊 EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata. 

😊 We use DIT and transformer as a diffuser for video and image generation.

😊 Welcome!

[![Arxiv Page](https://img.shields.io/badge/Arxiv-Page-red)](https://arxiv.org/abs/2405.18991)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://easyanimate.github.io/)
[![Modelscope Studio](https://img.shields.io/badge/Modelscope-Studio-blue)](https://modelscope.cn/studios/PAI/EasyAnimate/summary)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/EasyAnimate)
[![Discord Page](https://img.shields.io/badge/Discord-Page-blue)](https://discord.gg/UzkpB4Bn)

English | [简体中文](./README_zh-CN.md) | [日本語](./README_ja-JP.md)

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Video Result](#video-result)
- [How to use](#how-to-use)
- [Model zoo](#model-zoo)
- [TODO List](#todo-list)
- [Contact Us](#contact-us)
- [Reference](#reference)
- [License](#license)

# Introduction
EasyAnimate is a pipeline based on the transformer architecture, designed for generating AI images and videos, and for training baseline models and Lora models for Diffusion Transformer. We support direct prediction from pre-trained EasyAnimate models, allowing for the generation of videos with various resolutions, approximately 6 seconds in length, at 8fps (EasyAnimateV5, 1 to 49 frames). Additionally, users can train their own baseline and Lora models for specific style transformations.

We will support quick pull-ups from different platforms, refer to [Quick Start](#quick-start).

**New Features:**
- EasyAnimate-V5.1 is now supported in diffusers. For more implementation details, please refer to the [PR](https://github.com/huggingface/diffusers/pull/10626). Relevant weights can be downloaded from [EasyAnimate-V5.1-diffusers](https://huggingface.co/collections/alibaba-pai/easyanimate-v51-diffusers-67c81d1d19b236e056675cce). For usage instructions, please refer to [Usage](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers#a%E3%80%81text-to-video). [ 2025.03.06 ]
- **Updated to version v5.1**, the Qwen2 VL is used as the text encoder, and Flow is used as the sampling method. It supports bilingual prediction in both Chinese and English. In addition to common controls such as Canny and Pose, it also supports trajectory control, camera control. [2025.01.21]
- Use reward backpropagation to train Lora and optimize the video, aligning it better with human preferences, detailes in [here](scripts/README_TRAIN_REWARD.md). EasyAnimateV5-7b is released now. [2024.11.27]
- **Updated to v5**, supporting video generation up to 1024x1024, 49 frames, 6s, 8fps, with expanded model scale to 12B, incorporating the MMDIT structure, and enabling control models with diverse inputs; supports bilingual predictions in Chinese and English. [2024.11.08]
- **Updated to v4**, allowing for video generation up to 1024x1024, 144 frames, 6s, 24fps; supports video generation from text, image, and video, with a single model handling resolutions from 512 to 1280; bilingual predictions in Chinese and English enabled. [2024.08.15]
- **Updated to v3**, supporting video generation up to 960x960, 144 frames, 6s, 24fps, from text and image. [2024.07.01]
- **ModelScope-Sora “Data Director” Creative Race** — The third Data-Juicer Big Model Data Challenge is now officially launched! Utilizing EasyAnimate as the base model, it explores the impact of data processing on model training. Visit the [competition website](https://tianchi.aliyun.com/competition/entrance/532219) for details. [2024.06.17]
- **Updated to v2**, supporting video generation up to 768x768, 144 frames, 6s, 24fps. [2024.05.26]
- **Code Created!** Now supporting Windows and Linux. [2024.04.12]

Function：
- [Data Preprocessing](#data-preprocess)
- [Train VAE](#vae-train)
- [Train DiT](#dit-train)
- [Video Generation](#video-gen)

Our UI interface is as follows:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/ui_v3.jpg)

# Quick Start
### 1. Cloud usage: AliyunDSW/Docker
#### a. From AliyunDSW
DSW has free GPU time, which can be applied once by a user and is valid for 3 months after applying.

Aliyun provide free GPU time in [Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1), get it and use in Aliyun PAI-DSW to start EasyAnimate within 5min!

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/easyanimate_v5)

#### b. From ComfyUI
Our ComfyUI is as follows, please refer to [ComfyUI README](comfyui/README.md) for details.
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/comfyui_i2v.jpg)

#### c. From docker
If you are using docker, please make sure that the graphics card driver and CUDA environment have been installed correctly in your machine.

Then execute the following commands in this way:
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

# Please use the hugginface link or modelscope link to download the EasyAnimateV5.1 model.
# https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP
# https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-InP

# https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh
# https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh
```

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

The video size for EasyAnimateV5.1-12B can be generated by different GPU Memory, including:
| GPU memory | 384x672x25 | 384x672x49 | 576x1008x25 | 576x1008x49 | 768x1344x25 | 768x1344x49 |
|------------|------------|------------|------------|------------|------------|------------|
| 16GB | 🧡 | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ |
| 24GB | 🧡 | 🧡 | 🧡 | 🧡 | 🧡 | ❌ | 
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 

The video size for EasyAnimateV5.1-7B can be generated by different GPU Memory, including:
| GPU memory |384x672x25|384x672x49|576x1008x25|576x1008x49|768x1344x25|768x1344x49|
|----------|----------|----------|----------|----------|----------|----------|
| 16GB | 🧡 | 🧡 | ⭕️ | ⭕️ | ❌ | ❌ | 
| 24GB | ✅ | ✅ | ✅ | 🧡 | 🧡 | ❌ | 
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 

✅ indicates it can run under "model_cpu_offload", 🧡 represents it can run under "model_cpu_offload_and_qfloat8", ⭕️ indicates it can run under "sequential_cpu_offload", ❌ means it can't run. Please note that running with sequential_cpu_offload will be slower.

Some GPUs that do not support torch.bfloat16, such as 2080ti and V100, require changing the weight_dtype in app.py and predict files to torch.float16 in order to run.

The generation time for EasyAnimateV5.1-12B using different GPUs over 25 steps is as follows:

| GPU       | 384x672x25       | 384x672x49       | 576x1008x25      | 576x1008x49      | 768x1344x25      | 768x1344x49     |
|-----------|------------------|------------------|------------------|------------------|------------------|-----------------|
| A10 24GB  | ~120s (4.8s/it)  | ~240s (9.6s/it)  | ~320s (12.7s/it) | ~750s (29.8s/it) | ❌               | ❌              |
| A100 80GB | ~45s (1.75s/it)  | ~90s (3.7s/it)   | ~120s (4.7s/it)  | ~300s (11.4s/it) | ~265s (10.6s/it) | ~710s (28.3s/it) |

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>
  
The video size for EasyAnimateV3 can be generated by different GPU Memory, including:

| GPU memory | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 720x1280x72 | 720x1280x144 |
|------------|------------|-------------|-------------|--------------|-------------|--------------|
| 12GB       | ⭕️         | ⭕️          | ⭕️          | ⭕️           | ❌          | ❌           |
| 16GB       | ✅         | ✅          | ⭕️          | ⭕️           | ⭕️          | ❌           |
| 24GB       | ✅         | ✅          | ✅          | ✅           | ✅          | ❌           |
| 40GB       | ✅         | ✅          | ✅          | ✅           | ✅          | ✅           |
| 80GB       | ✅         | ✅          | ✅          | ✅           | ✅          | ✅           |

(⭕️) indicates it can run with low_gpu_memory_mode=True, but at a slower speed, and ❌ means it can't run.
</details>

#### b. Weights
We'd better place the [weights](#model-zoo) along the specified path:

EasyAnimateV5.1:
```
📦 models/
├── 📂 Diffusion_Transformer/
│   ├── 📂 EasyAnimateV5.1-12b-zh-InP/
│   └── 📂 EasyAnimateV5.1-12b-zh/
├── 📂 Personalized_Model/
│   └── your trained trainformer model / your trained lora model (for UI load)
```

# Video Result

### Image to Video with EasyAnimateV5.1-12b-zh-InP
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/74a23109-f555-4026-a3d8-1ac27bb3884c" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ab5aab27-fbd7-4f55-add9-29644125bde7" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/238043c2-cdbd-4288-9857-a273d96f021f" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/48881a0e-5513-4482-ae49-13a0ad7a2557" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3e7aba7f-6232-4f39-80a8-6cfae968f38c" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/986d9f77-8dc3-45fa-bc9d-8b26023fffbc" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/7f62795a-2b3b-4c14-aeb1-1230cb818067" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b581df84-ade1-4605-a7a8-fd735ce3e222" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/eab1db91-1082-4de2-bb0a-d97fd25ceea1" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/3fda0e96-c1a8-4186-9c4c-043e11420f05" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4b53145d-7e98-493a-83c9-4ea4f5b58289" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/75f7935f-17a8-4e20-b24c-b61479cf07fc" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### Text to Video with EasyAnimateV5.1-12b-zh

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/8818dae8-e329-4b08-94fa-00d923f38fd2" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d3e483c3-c710-47d2-9fac-89f732f2260a" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4dfa2067-d5d4-4741-a52c-97483de1050d" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fb44c2db-82c6-427e-9297-97dcce9a4948" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/dc6b8eaf-f21b-4576-a139-0e10438f20e4" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b3f8fd5b-c5c8-44ee-9b27-49105a08fbff" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/a68ed61b-eed3-41d2-b208-5f039bf2788e" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/4e33f512-0126-4412-9ae8-236ff08bcd21" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### Control Video with EasyAnimateV5.1-12b-zh-Control

Trajectory Control:
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bf3b8970-ca7b-447f-8301-72dfe028055b" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/63a7057b-573e-4f73-9d7b-8f8001245af4" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/090ac2f3-1a76-45cf-abe5-4e326113389b" width="100%" controls autoplay loop></video>
     </td>
  <tr>
</table>

Generic Control Video (Canny, Pose, Depth, etc.):
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/53002ce2-dd18-4d4f-8135-b6f68364cabd" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fce43c0b-81fa-4ab2-9ca7-78d786f520e6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/b208b92c-5add-4ece-a200-3dbbe47b93c3" width="100%" controls autoplay loop></video>
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3aec95d5-d240-49fb-a9e9-914446c7a4cf" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/60fa063b-5c1f-485f-b663-09bd6669de3f" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4adde728-8397-42f3-8a2a-23f7b39e9a1e" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### Camera Control with EasyAnimateV5.1-12b-zh-Control-Camera

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          Pan Up
      </td>
      <td>
          Pan Left
      </td>
       <td>
          Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/a88f81da-e263-4038-a5b3-77b26f79719e" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e346c59d-7bca-4253-97fb-8cbabc484afb" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4de470d4-47b7-46e3-82d3-b714a2f6aef6" width="100%" controls autoplay loop></video>
     </td>
  <tr>
      <td>
          Pan Down
      </td>
      <td>
          Pan Up + Pan Left
      </td>
       <td>
          Pan Up + Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7a3fecc2-d41a-4de3-86cd-5e19aea34a0d" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/cb281259-28b6-448e-a76f-643c3465672e" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/44faf5b6-d83c-4646-9436-971b2b9c7216" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

# How to use

<h3 id="video-gen">1. Inference </h3>

#### a. Memory-Saving Options
Since EasyAnimateV5 and V5.1 have very large parameters, we need to consider memory-saving options to adapt to consumer-grade graphics cards. We provide GPU_memory_mode for each prediction file, allowing you to choose from model_cpu_offload, model_cpu_offload_and_qfloat8, or sequential_cpu_offload.

- model_cpu_offload means the entire model will move to the CPU after use, saving some memory.
- model_cpu_offload_and_qfloat8 means the entire model will move to the CPU after use and applies float8 quantization to the transformer model, saving more memory.
- sequential_cpu_offload means each layer of the model moves to CPU after use, which is slower but saves a lot of memory.

qfloat8 may reduce model performance but saves more memory. If memory is sufficient, it's recommended to use model_cpu_offload.

#### b. Via ComfyUI
For more details, see the [ComfyUI README](comfyui/README.md).

#### c. Run Python Files
- Step 1: Download the corresponding [weights](#model-zoo) and place them in the models folder.
- Step 2: Use different files for predictions based on the weights and prediction goals.
  - Text-to-Video:
    - Modify the prompt, neg_prompt, guidance_scale, and seed in the predict_t2v.py file.
    - Then run the predict_t2v.py file and wait for the results, which are stored in the samples/easyanimate-videos folder.
  - Image-to-Video:
    - Modify validation_image_start, validation_image_end, prompt, neg_prompt, guidance_scale, and seed in the predict_i2v.py file.
    - validation_image_start is the starting image, and validation_image_end is the ending image of the video.
    - Then run the predict_i2v.py file and wait for the results, which are stored in the samples/easyanimate-videos_i2v folder.
  - Video-to-Video:
    - Modify validation_video, validation_image_end, prompt, neg_prompt, guidance_scale, and seed in the predict_v2v.py file.
    - validation_video is the reference video for video-to-video. You can run a demo with the following video: [Demo Video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4)
    - Then run the predict_v2v.py file and wait for the results, which are stored in samples/easyanimate-videos_v2v folder.
  - Generic Control Video (Canny, Pose, Depth, etc.):
    - Modify control_video, validation_image_end, prompt, neg_prompt, guidance_scale, and seed in the predict_v2v_control.py file.
    - control_video is the control video for video generation, extracted using Canny, Pose, Depth, etc. You can run a demo with the following video: [Demo Video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4)
    - Then run the predict_v2v_control.py file and wait for the results, which are stored in samples/easyanimate-videos_v2v_control folder.
  - Trajectory Control Video:
    - Modify control_video, ref_image, validation_image_end, prompt, neg_prompt, guidance_scale, and seed in the predict_v2v_control.py file.
    - control_video is the control video, and ref_image is the reference first frame image. You can run a demo with the following image and video: [Demo Image](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/dog.png), [Demo Video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/trajectory_demo.mp4)
    - Then run the predict_v2v_control.py file and wait for the results, which are stored in samples/easyanimate-videos_v2v_control folder.
    - Interaction via ComfyUI is recommended.
  - Camera Control Video:
    - Modify control_video, ref_image, validation_image_end, prompt, neg_prompt, guidance_scale, and seed in the predict_v2v_control.py file.
    - control_camera_txt is the control file for camera control video, and ref_image is the reference first frame image. You can run a demo with the following image and control file: [Demo Image](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/firework.png), [Demo File (from CameraCtrl)](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/0a3b5fb184936a83.txt)
    - Then run the predict_v2v_control.py file and wait for the results, which are stored in samples/easyanimate-videos_v2v_control folder.
    - Interaction via ComfyUI is recommended.
- Step 3: To combine with other backbones and Lora trained by yourself, modify predict_t2v.py and lora_path accordingly in the predict_t2v.py file.

#### d. Via WebUI Interface

WebUI supports text-to-video, image-to-video, video-to-video, and control-based video generation (such as Canny, Pose, Depth, etc.).

- Step 1: Download the corresponding [weights](#model-zoo) and place them in the models folder.
- Step 2: Run the app.py file to enter the Gradio page.
- Step 3: Choose the generation model from the page, fill in prompt, neg_prompt, guidance_scale, seed, etc., click generate, and wait for the results, which are stored in the sample folder.


### 2. Model Training
A complete EasyAnimate training pipeline should include data preprocessing, Video VAE training, and Video DiT training. Among these, Video VAE training is optional because we have already provided a pre-trained Video VAE.

<h4 id="data-preprocess">a. data preprocessing</h4>

We provide two simple demos:
- Train a Lora model using image data. For more details, you can refer to the [wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-Lora).
- Perform SFT model training using video data. For more details, you can refer to the [wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-SFT).

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

EasyAnimateV5.1:

7B:
| Name | Type | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|--|
| EasyAnimateV5.1-7b-zh-InP | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh-InP) | Official image-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |
| EasyAnimateV5.1-7b-zh-Control | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-Control) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh-Control) | Official video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, and trajectory control. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |
| EasyAnimateV5.1-7b-zh-Control-Camera | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh-Control-Camera) | Official video camera control weights, supporting direction generation control by inputting camera motion trajectories. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |
| EasyAnimateV5.1-7b-zh | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh) | Official text-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |

12B:
| Name | Type | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|--|
| EasyAnimateV5.1-12b-zh-InP | EasyAnimateV5.1 | 39 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-InP) | Official image-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |
| EasyAnimateV5.1-12b-zh-Control | EasyAnimateV5.1 | 39 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-Control) | Official video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, and trajectory control. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |
| EasyAnimateV5.1-12b-zh-Control-Camera | EasyAnimateV5.1 | 39 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-Control-Camera) | Official video camera control weights, supporting direction generation control by inputting camera motion trajectories. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |
| EasyAnimateV5.1-12b-zh | EasyAnimateV5.1 | 39 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh) | Official text-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports for multilingual prediction. |

<details>
  <summary>(Obsolete) EasyAnimateV5:</summary>

7B:
| Name | Type | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|--|
| EasyAnimateV5-7b-zh-InP | EasyAnimateV5 | 22 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-7b-zh-InP) | Official 7B image-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports bilingual prediction in Chinese and English. |
| EasyAnimateV5-7b-zh | EasyAnimateV5 | 22 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-7b-zh) | Official 7B text-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports bilingual prediction in Chinese and English. |
| EasyAnimateV5-Reward-LoRAs | EasyAnimateV5 | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-Reward-LoRAs) | The official reward backpropagation technology model optimizes the videos generated by EasyAnimateV5-12b to better match human preferences. ｜

12B:
| Name | Type | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|--|
| EasyAnimateV5-12b-zh-InP | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh-InP) | Official image-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports bilingual prediction in Chinese and English. |
| EasyAnimateV5-12b-zh-Control | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-Control) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh-Control) | Official video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc. Supports video prediction at multiple resolutions (512, 768, 1024) and is trained with 49 frames at 8 frames per second. Bilingual prediction in Chinese and English is supported. |
| EasyAnimateV5-12b-zh | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh) | Official text-to-video weights. Supports video prediction at multiple resolutions (512, 768, 1024), trained with 49 frames at 8 frames per second, and supports bilingual prediction in Chinese and English. |
| EasyAnimateV5-Reward-LoRAs | EasyAnimateV5 | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-Reward-LoRAs) | The official reward backpropagation technology model optimizes the videos generated by EasyAnimateV5-12b to better match human preferences. ｜
</details>

<details>
  <summary>(Obsolete) EasyAnimateV4:</summary>

| Name | Type | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|--|
| EasyAnimateV4-XL-2-InP | EasyAnimateV4 | Before extraction: 8.9 GB \/ After extraction: 14.0 GB |[🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV4-XL-2-InP)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV4-XL-2-InP)| | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 144 frames at a rate of 24 frames per second. |
</details>

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>

| Name | Type | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|--|
| EasyAnimateV3-XL-2-InP-512x512 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-512x512)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-512x512) | EasyAnimateV3 official weights for 512x512 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-768x768 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-768x768) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-768x768) | EasyAnimateV3 official weights for 768x768 text and image to video resolution. Training with 144 frames and fps 24 |
| EasyAnimateV3-XL-2-InP-960x960 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-960x960) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-960x960) | EasyAnimateV3 official weights for 960x960 text and  image to video resolution. Training with 144 frames and fps 24 |
</details>

<details>
  <summary>(Obsolete) EasyAnimateV2:</summary>

| Name | Type | Storage Space | Url | Hugging Face | Model Scope | Description |
|--|--|--|--|--|--|--|
| EasyAnimateV2-XL-2-512x512 | EasyAnimateV2 | 16.2GB | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-512x512)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV2-XL-2-512x512)| EasyAnimateV2 official weights for 512x512 resolution. Training with 144 frames and fps 24 |
| EasyAnimateV2-XL-2-768x768 | EasyAnimateV2 | 16.2GB | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-768x768) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV2-XL-2-768x768)| EasyAnimateV2 official weights for 768x768 resolution. Training with 144 frames and fps 24 |
| easyanimatev2_minimalism_lora.safetensors | Lora of Pixart | 485.1MB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimatev2_minimalism_lora.safetensors)| - | - | A lora training with a specifial type images. Images can be downloaded from [Url](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v2/Minimalism.zip). |
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

# TODO List
- Support model with larger params.

# Contact Us
1. Use Dingding to search group 77450006752 or Scan to join
2. You need to scan the image to join the WeChat group or if it is expired, add this student as a friend first to invite you.

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/dd.png" alt="ding group" width="30%"/>
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/wechat.jpg" alt="Wechat group" width="30%"/>
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/person.jpg" alt="Person" width="30%"/>


# Reference
- CogVideo: https://github.com/THUDM/CogVideo/
- Flux: https://github.com/black-forest-labs/flux
- magvit: https://github.com/google-research/magvit
- PixArt: https://github.com/PixArt-alpha/PixArt-alpha
- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- Animatediff: https://github.com/guoyww/AnimateDiff
- HunYuan DiT: https://github.com/tencent/HunyuanDiT
- ComfyUI-KJNodes: https://github.com/kijai/ComfyUI-KJNodes
- ComfyUI-EasyAnimateWrapper: https://github.com/kijai/ComfyUI-EasyAnimateWrapper
- ComfyUI-CameraCtrl-Wrapper: https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper
- CameraCtrl: https://github.com/hehao13/CameraCtrl
- DragAnything: https://github.com/showlab/DragAnything

# License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
