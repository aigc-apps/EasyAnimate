# EasyAnimate | 视频与图片一体化生成基线方案。
😊 EasyAnimate是一个用于生成长视频和图片 和 训练基于transformer的扩散生成器的repo。

😊 我们基于类SORA结构与DIT，使用transformer进行作为扩散器进行视频与图片生成。为了保证良好的拓展性，我们基于motion module构建了EasyAnimate，未来我们也会尝试更多的训练方案一提高效果。

😊 Welcome!

[English](./README.md) | 简体中文

# 目录
- [EasyPhoto | 您的智能生成器。](#easyphoto--您的智能生成器)
- [目录](#目录)
- [简介](#简介)
- [TODO List](#todo-list)
- [Model zoo](#model-zoo)
- [快速启动](#快速启动)
- [如何使用](#如何使用)
- [算法细节](#算法细节)
- [参考文献](#参考文献)
- [许可证](#许可证)

# 简介
EasyAnimate是一个基于transformer结构的pipeline，可用于生成AI图片与视频、训练Diffusion Transformer的基线模型与Lora模型，我们支持从已经训练好的EasyAnimate模型直接进行预测，生成不同分辨率，6秒左右、fps24的视频（1 ~ 144帧, 未来会支持更长的视频），也支持用户训练自己的基线模型与Lora模型，进行一定的风格变换。

我们会逐渐支持从不同平台快速启动，请参阅 [快速启动](#快速启动)。

新特性：
- 更新到v2版本，最大支持144帧(6s, 24fps)生成。[ 2024.05.26 ]
- 创建代码！现在支持 Windows 和 Linux。[ 2024.04.12 ]

这些是我们的生成结果:
![Combine_512](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v2/Combine_512.jpg)

我们的ui界面如下:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/ui.png)

# TODO List
- 支持更大分辨率的文视频生成模型。
- 支持视频inpaint模型。

# Model zoo

EasyAnimateV2:
| 名称 | 种类 | 存储空间 | 下载地址 | 描述 |
|--|--|--|--|--| 
| EasyAnimateV2-XL-2-512x512.tar | EasyAnimateV2 | 16.2GB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV2-XL-2-512x512.tar)| EasyAnimateV2 official weights for 512x512 resolution. Training with 144 frames and fps 24 |
| EasyAnimateV2-XL-2-768x768.tar | EasyAnimateV2 | 16.2GB | Coming soon | EasyAnimateV2 official weights for 768x768 resolution. Training with 144 frames and fps 24 |
| easyanimatev2_minimalism_lora.safetensors | Lora of Pixart | 654.0MB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimatev2_minimalism_lora.safetensors)| A lora training with a specifial type images. Images can be downloaded from [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/Minimalism.zip). |

<details>
  <summary>EasyAnimateV1:</summary>

### 1、运动权重
| 名称 | 种类 | 存储空间 | 下载地址 | 描述 |
|--|--|--|--|--| 
| easyanimate_v1_mm.safetensors | Motion Module | 4.1GB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Motion_Module/easyanimate_v1_mm.safetensors) | Training with 80 frames and fps 12 |

### 2、其他权重
| 名称 | 种类 | 存储空间 | 下载地址 | 描述 |
|--|--|--|--|--| 
| PixArt-XL-2-512x512.tar | Pixart | 11.4GB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/PixArt-XL-2-512x512.tar)| Pixart-Alpha official weights |
| easyanimate_portrait.safetensors | Checkpoint of Pixart | 2.3GB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait.safetensors) | Training with internal portrait datasets |
| easyanimate_portrait_lora.safetensors | Lora of Pixart | 654.0MB | [download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait_lora.safetensors)| Training with internal portrait datasets |
</details>

# 生成效果
我们将一些结果展示在这里 [GALLERY](scripts/Result%20Gallery.md). 

# 快速启动
### 1. 云使用: AliyunDSW/Docker
#### a. 通过阿里云 DSW
敬请期待。

#### b. 通过docker
使用docker的情况下，请保证机器中已经正确安装显卡驱动与CUDA环境，然后以此执行以下命令：

EasyAnimateV2: 
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

<details>
  <summary>EasyAnimateV1:</summary>
  
```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# 进入镜像
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# clone 代码
git clone https://github.com/aigc-apps/EasyAnimate.git

# 进入EasyAnimate文件夹
cd EasyAnimate

# 下载权重
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Motion_Module/easyanimate_v1_mm.safetensors -O models/Motion_Module/easyanimate_v1_mm.safetensors
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait.safetensors -O models/Personalized_Model/easyanimate_portrait.safetensors
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait_lora.safetensors -O models/Personalized_Model/easyanimate_portrait_lora.safetensors
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/PixArt-XL-2-512x512.tar -O models/Diffusion_Transformer/PixArt-XL-2-512x512.tar

cd models/Diffusion_Transformer/
tar -xvf PixArt-XL-2-512x512.tar
cd ../../
```
</details>

### 2. 本地安装: 环境检查/下载/安装
#### a. 环境检查
我们已验证EasyPhoto可在以下环境中执行：

Linux 的详细信息：
- 操作系统 Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8
- CUDNN: 8+
- GPU： Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

我们需要大约 60GB 的可用磁盘空间，请检查！

#### b. 权重放置
我们最好将权重按照指定路径进行放置：

EasyAnimateV2: 
```
📦 models/
├── 📂 Diffusion_Transformer/
│   └── 📂 EasyAnimateV2-XL-2-512x512/
```

<details>
  <summary>EasyAnimateV1:</summary>

  ```
  📦 models/
  ├── 📂 Diffusion_Transformer/
  │   └── 📂 PixArt-XL-2-512x512/
  ├── 📂 Motion_Module/
  │   └── 📄 easyanimate_v1_mm.safetensors
  ├── 📂 Motion_Module/
  │   ├── 📄 easyanimate_portrait.safetensors
  │   └── 📄 easyanimate_portrait_lora.safetensors
  ```
</details>

# 如何使用
### 1. 生成
#### a. 视频生成
##### i、运行python文件
- 步骤1：下载对应权重放入models文件夹。
- 步骤2：在predict_t2v.py文件中修改prompt、neg_prompt、guidance_scale和seed。
- 步骤3：运行predict_t2v.py文件，等待生成结果，结果保存在samples/easyanimate-videos文件夹中。
- 步骤4：如果想结合自己训练的其他backbone与Lora，则看情况修改predict_t2v.py中的predict_t2v.py和lora_path。

##### ii、通过ui界面
- 步骤1：下载对应权重放入models文件夹。
- 步骤2：运行app.py文件，进入gradio页面。
- 步骤3：根据页面选择生成模型，填入prompt、neg_prompt、guidance_scale和seed等，点击生成，等待生成结果，结果保存在sample文件夹中。

### 2. 模型训练
如果期望训练一个文生图视频的生成模型，您需要以这种格式排列数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000001.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.json是一个标准的json文件，如下所示：
```json
[
    {
      "file_path": "videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```
json中的file_path可以被设置为相对路径。

然后，进入scripts/train_t2iv.sh进行设置。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

你也可以将路径设置为绝对路径：
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
此时 scripts/train_t2iv.sh 可以被设置为如下:
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

最后运行scripts/train_t2iv.sh。
```sh
sh scripts/train_t2iv.sh
```

# 算法细节
我们使用了[PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)作为基础模型，并在此基础上引入额外的运动模块（motion module）来将DiT模型从2D图像生成扩展到3D视频生成上来。其框架图如下：

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/pipeline.png" alt="ui" style="zoom:50%;" />

其中，Motion Module 用于捕捉时序维度的帧间关系，其结构如下：

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/motion_module.png" alt="motion" style="zoom:50%;" />

我们在时序维度上引入注意力机制来让模型学习时序信息，以进行连续视频帧的生成。同时，我们利用额外的网格计算（Grid Reshape），来扩大注意力机制的input token数目，从而更多地利用图像的空间信息以达到更好的生成效果。Motion Module 作为一个单独的模块，在推理时可以用在不同的DiT基线模型上。此外，EasyAnimate不仅支持了motion-module模块的训练，也支持了DiT基模型/LoRA模型的训练，以方便用户根据自身需要来完成自定义风格的模型训练，进而生成任意风格的视频。

# 参考文献
- magvit: https://github.com/google-research/magvit
- PixArt: https://github.com/PixArt-alpha/PixArt-alpha
- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- Animatediff: https://github.com/guoyww/AnimateDiff

# 许可证
本项目采用 [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
