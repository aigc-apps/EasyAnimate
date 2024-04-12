# EasyAnimate | 您的智能生成器。
😊 EasyAnimate是一个用于生成长视频和训练基于transformer的扩散生成器的repo。

😊 Welcome!

[English](./README.md) | 简体中文

# 目录
- [EasyPhoto | 您的智能生成器。](#easyphoto--您的智能生成器)
- [目录](#目录)
- [简介](#简介)
- [TODO List](#todo-list)
- [Model zoo](#model-zoo)
    - [1、运动权重](#1运动权重)
    - [2、其他权重](#2其他权重)
- [快速启动](#快速启动)
    - [1. 云使用: AliyunDSW/Docker](#1-云使用-aliyundswdocker)
    - [2. 本地安装: 环境检查/下载/安装](#2-本地安装-环境检查下载安装)
- [如何使用](#如何使用)
    - [1. 生成](#1-生成)
    - [2. 模型训练](#2-模型训练)
- [算法细节](#算法细节)
- [参考文献](#参考文献)
- [许可证](#许可证)

# 简介
EasyAnimate是一个基于transformer结构的pipeline，可用于生成AI动画、训练Diffusion Transformer的基线模型与Lora模型，我们支持从已经训练好的EasyAnimate模型直接进行预测，生成5秒左右、fps12的视频（未来会支持更长的视频），也支持用户训练自己的基线模型与Lora模型，进行一定的风格变换。

我们会逐渐支持从不同平台快速启动，请参阅 [快速启动](#快速启动)。

新特性：
- 创建代码！现在支持 Windows 和 Linux。[ 2024.04.12 ]

这些是我们的生成结果:

我们的ui界面如下:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/ui.png)

# TODO List
- 支持不同分辨率的文图生成模型。
- 支持不同帧数的文图生成模型。
- 支持基于magvit的文图生成模型。
- 支持视频inpaint模型。

# Model zoo
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


# 生成效果
在生成风景类animation时，采样器推荐使用DPM++和Euler A。在生成人像类animation时，采样器推荐使用Euler A和Euler。

有些时候Github无法正常显示大GIF，可以通过Download GIF下载到本地查看。

| A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest. | The video captures the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird\'s eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature\'s power and beauty. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/star.gif) [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/star.gif) | ![00000002](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/waterfull.gif) [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/waterfull.gif) |
| **A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff\'s precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures.** | **1girl, bangs, blue eyes, blunt bangs, 4k, best quality, blurry, blurry background, bob cut, depth of field, lips, looking at viewer, motion blur, nose, realistic, red lips, shirt, short hair, solo, white shirt, (best quality), (realistic, photo-realistic:1.3), (beautiful eyes:1.3), (sparkling eyes:1.3), (beautiful mouth:1.3), finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup. （基于人像的 DiT）** |
| ![00000003](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/water.gif) [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/water.gif) | ![00000004](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/girl.gif) [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/girl.gif) |
| **A serene sunrise over the Grand Canyon, with elegant condors soaring in the morning breeze.** | **A tranquil Vermont autumn, with leaves in vibrant colors of orange and red fluttering down a mountain stream.** |
| ![00000005](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/sunshine.gif) [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/sunshine.gif)| ![00000006](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/autumn.gif) [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/autumn.gif)|

# 快速启动
### 1. 云使用: AliyunDSW/Docker
#### a. 通过阿里云 DSW
敬请期待。

#### b. 通过docker
使用docker的情况下，请保证机器中已经正确安装显卡驱动与CUDA环境，然后以此执行以下命令：
```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# 进入镜像
docker run -it -p 7860:7860 --network host --gpus all mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

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
#### a、训练视频生成模型
##### i、基于webvid数据集
如果使用webvid数据集进行训练，则需要首先下载webvid的数据集。

您需要以这种格式排列webvid数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 webvid/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.mp4
│       │   └── 📄 .....
│       └── 📄 csv_of_webvid.csv
```

然后，进入scripts/train_t2v.sh进行设置。
```
export DATASET_NAME="datasets/webvid/videos/"
export DATASET_META_NAME="datasets/webvid/csv_of_webvid.csv"

...

train_data_format="webvid"
```

最后运行scripts/train_t2v.sh。
```sh
sh scripts/train_t2v.sh
```

##### ii、基于自建数据集
如果使用内部数据集进行训练，则需要首先格式化数据集。

您需要以这种格式排列数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.mp4
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
      "file_path": "videos/00000002.mp4",
      "text": "A notepad with a drawing of a woman on it.",
      "type": "video"
    }
    .....
]
```
json中的file_path需要设置为相对路径。

然后，进入scripts/train_t2v.sh进行设置。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

最后运行scripts/train_t2v.sh。
```sh
sh scripts/train_t2v.sh
```

#### b、训练基础文生图模型
##### i、基于diffusers格式
数据集的格式可以设置为diffusers格式。

```
📦 project/
├── 📂 datasets/
│   ├── 📂 diffusers_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 metadata.jsonl
```

然后，进入scripts/train_t2i.sh进行设置。
```
export DATASET_NAME="datasets/diffusers_datasets/"

...

train_data_format="diffusers"
```

最后运行scripts/train_t2i.sh。
```sh
sh scripts/train_t2i.sh
```
##### ii、基于自建数据集
如果使用自建数据集进行训练，则需要首先格式化数据集。

您需要以这种格式排列数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.json是一个标准的json文件，如下所示：
```json
[
    {
      "file_path": "train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A notepad with a drawing of a woman on it.",
      "type": "image"
    }
    .....
]
```
json中的file_path需要设置为相对路径。

然后，进入scripts/train_t2i.sh进行设置。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

最后运行scripts/train_t2i.sh。
```sh
sh scripts/train_t2i.sh
```

#### c、训练Lora模型
##### i、基于diffusers格式
数据集的格式可以设置为diffusers格式。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 diffusers_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 metadata.jsonl
```

然后，进入scripts/train_lora.sh进行设置。
```
export DATASET_NAME="datasets/diffusers_datasets/"

...

train_data_format="diffusers"
```

最后运行scripts/train_lora.sh。
```sh
sh scripts/train_lora.sh
```

##### ii、基于自建数据集
如果使用自建数据集进行训练，则需要首先格式化数据集。

您需要以这种格式排列数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.json是一个标准的json文件，如下所示：
```json
[
    {
      "file_path": "train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A notepad with a drawing of a woman on it.",
      "type": "image"
    }
    .....
]
```
json中的file_path需要设置为相对路径。

然后，进入scripts/train_lora.sh进行设置。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

最后运行scripts/train_lora.sh。
```sh
sh scripts/train_lora.sh
```
# 算法细节
我们使用了[PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)作为基础模型，并在此基础上引入额外的运动模块（motion module）来将DiT模型从2D图像生成扩展到3D视频生成上来。其框架图如下：



<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/pipeline.png" alt="ui" style="zoom:50%;" />



其中，Motion Module 用于捕捉时序维度的帧间关系，其结构如下：



<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/motion_module.png" alt="motion" style="zoom:50%;" />



我们在时序维度上引入注意力机制来让模型学习时序信息，以进行连续视频帧的生成。同时，我们利用额外的网格计算（Grid Reshape），来扩大注意力机制的input token数目，从而更多地利用图像的空间信息以达到更好的生成效果。Motion Module 作为一个单独的模块，在推理时可以用在不同的DiT基线模型上。此外，EasyAnimate不仅支持了motion-module模块的训练，也支持了DiT基模型/LoRA模型的训练，以方便用户根据自身需要来完成自定义风格的模型训练，进而生成任意风格的视频。


# 算法限制
- 受

# 参考文献
- magvit: https://github.com/google-research/magvit
- PixArt: https://github.com/PixArt-alpha/PixArt-alpha
- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- Animatediff: https://github.com/guoyww/AnimateDiff

# 许可证
本项目采用 [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
