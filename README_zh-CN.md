# EasyAnimate | 您的智能生成器。
😊 EasyAnimate是一个用于生成长视频和训练基于transformer的扩散生成器的repo。

😊 我们基于类SORA结构与DIT，使用transformer进行作为扩散器进行视频生成。为了保证良好的拓展性，我们基于motion module构建了EasyAnimate，未来我们也会尝试更多的训练方案一提高效果。

😊 Welcome!

[English](./README.md) | 简体中文

# 目录
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
EasyAnimate是一个基于transformer结构的pipeline，可用于生成AI动画、训练Diffusion Transformer的基线模型与Lora模型，我们支持从已经训练好的EasyAnimate模型直接进行预测，生成不同分辨率，6秒左右、fps12的视频（40 ~ 80帧, 未来会支持更长的视频），也支持用户训练自己的基线模型与Lora模型，进行一定的风格变换。

我们会逐渐支持从不同平台快速启动，请参阅 [快速启动](#快速启动)。

新特性：
- 创建代码！现在支持 Windows 和 Linux。[ 2024.04.12 ]

这些是我们的生成结果:

我们的ui界面如下:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/ui.png)

# TODO List
- 支持更大分辨率的文视频生成模型。
- 支持基于magvit的文视频生成模型。
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

使用原始的pixart checkpoint进行预测。

| Base Models | Sampler | Seed | Resolution (h x w x f) | Prompt | GenerationResult | Download | 
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| PixArt | DPM++ | 43 | 512x512x80 | A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff\'s precipices. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/1-cliff.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/1-cliff.gif) |
| PixArt | DPM++ | 43 | 448x640x80 | The video captures the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/2-waterfall.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/2-waterfall.gif) |
| PixArt | DPM++ | 43 | 704x384x80 | A vibrant scene of a snowy mountain landscape. The sky is filled with a multitude of colorful hot air balloons, each floating at different heights, creating a dynamic and lively atmosphere. The balloons are scattered across the sky, some closer to the viewer, others further away, adding depth to the scene. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/3-snowy.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/3-snowy.gif) |
| PixArt | DPM++ | 43 | 448x640x64 | The vibrant beauty of a sunflower field. The sunflowers, with their bright yellow petals and dark brown centers, are in full bloom, creating a stunning contrast against the green leaves and stems. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/4-sunflower.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/4-sunflower.gif) |
| PixArt | DPM++ | 43 | 384x704x48 | A tranquil Vermont autumn, with leaves in vibrant colors of orange and red fluttering down a mountain stream. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/5-autumn.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/5-autumn.gif) |
| PixArt | DPM++ | 43 | 704x384x48 | A vibrant underwater scene. A group of blue fish, with yellow fins, are swimming around a coral reef. The coral reef is a mix of brown and green, providing a natural habitat for the fish. The water is a deep blue, indicating a depth of around 30 feet. The fish are swimming in a circular pattern around the coral reef, indicating a sense of motion and activity. The overall scene is a beautiful representation of marine life. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/6-underwater.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/6-underwater.gif) |
| PixArt | DPM++ | 43 | 576x448x48 | Pacific coast, carmel by the blue sea ocean and peaceful waves | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/7-coast.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/7-coast.gif) |
| PixArt | DPM++ | 43 | 576x448x80 | A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. The style of the video is a natural landscape shot, with a focus on the beauty of the snowy forest and the peacefulness of the road. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/8-forest.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/8-forest.gif) |
| PixArt | DPM++ | 43 | 640x448x64 | The dynamic movement of tall, wispy grasses swaying in the wind. The sky above is filled with clouds, creating a dramatic backdrop. The sunlight pierces through the clouds, casting a warm glow on the scene. The grasses are a mix of green and brown, indicating a change in seasons. The overall style of the video is naturalistic, capturing the beauty of the landscape in a realistic manner. The focus is on the grasses and their movement, with the sky serving as a secondary element. The video does not contain any human or animal elements. |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/9-grasses.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/9-grasses.gif) |
| PixArt | DPM++ | 43 | 704x384x80 | A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest. |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/10-night.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/10-night.gif) |
| PixArt | DPM++ | 43 | 640x448x80 | Sunset over the sea. | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/11-sunset.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/11-sunset.gif) |

使用人像checkpoint进行预测。

| Base Models | Sampler | Seed | Resolution (h x w x f) | Prompt | GenerationResult | Download | 
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Portrait | Euler A | 43 | 448x576x80 | 1girl, 3d, black hair, brown eyes, earrings, grey background, jewelry, lips, long hair, looking at viewer, photo \\(medium\\), realistic, red lips, solo | ![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/1-check.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/1-check.gif) |
| Portrait | Euler A | 43 | 448x576x80 | 1girl, bare shoulders, blurry, brown eyes, dirty, dirty face, freckles, lips, long hair, looking at viewer, realistic, sleeveless, solo, upper body |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/2-check.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/2-check.gif) |
| Portrait | Euler A | 43 | 512x512x64 | 1girl, black hair, brown eyes, earrings, grey background, jewelry, lips, looking at viewer, mole, mole under eye, neck tattoo, nose, ponytail, realistic, shirt, simple background, solo, tattoo |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/3-check.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/3-check.gif) |
| Portrait | Euler A | 43 | 576x448x64 | 1girl, black hair, lips, looking at viewer, mole, mole under eye, mole under mouth, realistic, solo |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/5-check.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/5-check.gif) |

使用人像Lora进行预测。

| Base Models | Sampler | Seed | Resolution (h x w x f) | Prompt | GenerationResult | Download | 
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Pixart + Lora | Euler A | 43 | 512x512x64 | 1girl, 3d, black hair, brown eyes, earrings, grey background, jewelry, lips, long hair, looking at viewer, photo \\(medium\\), realistic, red lips, solo |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/1-lora.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/1-lora.gif) |
| Pixart + Lora | Euler A | 43 | 512x512x64 | 1girl, bare shoulders, blurry, brown eyes, dirty, dirty face, freckles, lips, long hair, looking at viewer, mole, mole on breast, mole on neck, mole under eye, mole under mouth, realistic, sleeveless, solo, upper body |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/2-lora.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/2-lora.gif) |
| Pixart + Lora | Euler A | 43 | 512x512x64 | 1girl, black hair, lips, looking at viewer, mole, mole under eye, mole under mouth, realistic, solo |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/5-lora.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/5-lora.gif) |
| Pixart + Lora | Euler A | 43 | 512x512x80 | 1girl, bare shoulders, blurry, blurry background, blurry foreground, bokeh, brown eyes, christmas tree, closed mouth, collarbone, depth of field, earrings, jewelry, lips, long hair, looking at viewer, photo \\(medium\\), realistic, smile, solo |![00000001](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/low_resolution/8-lora.gif) | [Download GIF](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/8-lora.gif) |

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

### 2. 本地安装: 环境检查/下载/安装
#### a. 环境检查
我们已验证EasyAnimate可在以下环境中执行：

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
