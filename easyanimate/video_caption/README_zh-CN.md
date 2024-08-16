# 数据预处理
[English](./README.md) | 简体中文

该文件夹包含 EasyAnimate 使用的数据集预处理（即视频切分）、过滤和生成描述的代码。整个过程支持分布式并行处理，能够处理大规模数据集。

此外，我们和 [Data-Juicer](https://github.com/modelscope/data-juicer/blob/main/docs/DJ_SORA.md) 合作，能让你在 [Aliyun PAI-DLC](https://help.aliyun.com/zh/pai/user-guide/video-preprocessing/) 轻松进行视频数据的处理。


## 快速开始

### 安装
推荐使用阿里云 DSW 和 Docker 来安装环境，请参考 [快速开始](../../README_zh-CN.md#1-云使用-aliyundswdocker). 你也可以参考 [Dockerfile](../../Dockerfile.ds) 中的镜像构建流程在本地安装对应的 conda 环境和其余依赖。

为了提高推理速度和节省推理的显存，生成视频描述依赖于 [llm-awq](https://github.com/mit-han-lab/llm-awq)。因此，需要 RTX 3060 或者 A2 及以上的显卡 (CUDA Compute Capability >= 8.0)。

```shell
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:
asyanimate

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:asyanimate

# clone code
git clone https://github.com/aigc-apps/EasyAnimate.git

# enter video_caption
cd EasyAnimate/easyanimate/video_caption
```

### 数据准备
将下载的视频准备到文件夹 [datasets](./datasets/)（最好不使用嵌套结构，因为视频名称在后续处理中用作唯一 ID）。以 Panda-70M 为例，完整的数据集目录结构如下所示：
```
📦 datasets/
├── 📂 panda_70m/
│   ├── 📂 videos/
│   │   ├── 📂 data/
│   │   │   └── 📄 --C66yU3LjM_2.mp4
│   │   │   └── 📄 ...
```

### 视频切分
EasyAnimate 使用 [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) 来识别视频中的场景变化
并根据某些阈值通过 FFmpeg 执行视频分割，以确保视频片段的一致性。
短于 3 秒的视频片段将被丢弃，长于 10 秒的视频片段将被递归切分。

视频切分的完整流程在 [stage_1_video_splitting.sh](./scripts/stage_1_video_splitting.sh)。执行
```shell
sh scripts/stage_1_video_splitting.sh
```
后，切分后的视频位于 `easyanimate/video_caption/datasets/panda_70m/videos_clips/data/`。

### 视频过滤
基于上一步获得的视频，EasyAnimate 提供了一个简单而有效的流程来过滤出高质量的视频。总体流程如下：

- 美学过滤：通过 [aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5) 计算均匀采样的 4 帧视频的平均美学分数，从而筛选出内容不佳（模糊、昏暗等）的视频。
- 文本过滤：使用 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 计算中间帧的文本区域比例，过滤掉含有大面积文本的视频。
- 运动过滤：计算帧间光流差，过滤掉移动太慢或太快的视频。

视频过滤的完整流程在 [stage_2_video_filtering.sh](./scripts/stage_2_video_filtering.sh)。执行
```shell
sh scripts/stage_2_video_filtering.sh
```
后，视频的美学得分、文本得分和运动得分对应的元文件保存在 `easyanimate/video_caption/datasets/panda_70m/videos_clips/`。

> [!NOTE]
> 美学得分的计算依赖于 [google/siglip-so400m-patch14-384 model](https://huggingface.co/google/siglip-so400m-patch14-384).
请执行 `HF_ENDPOINT=https://hf-mirror.com sh scripts/stage_2_video_filtering.sh` 如果你无法访问 huggingface.com.

### 视频描述
在获得上述高质量的过滤视频后，EasyAnimate 利用 [VILA1.5](https://github.com/NVlabs/VILA) 来生成视频描述。随后，使用 LLMs 对生成的视频描述进行重写，以更好地满足视频生成任务的要求。最后，使用自研的 VideoCLIPXL 模型来过滤掉描述和视频内容不一致的数据，从而得到最终的训练数据集。

请根据机器的显存从 [VILA1.5](https://huggingface.co/collections/Efficient-Large-Model/vila-on-pre-training-for-visual-language-models-65d8022a3a52cd9bcd62698e) 下载合适大小的模型。对于 A100 40G，你可以执行下面的命令来下载 [VILA1.5-40b-AWQ](https://huggingface.co/Efficient-Large-Model/VILA1.5-40b-AWQ)
```shell
# Add HF_ENDPOINT=https://hf-mirror.com before the command if you cannot access to huggingface.com
huggingface-cli download Efficient-Large-Model/VILA1.5-40b-AWQ --local-dir-use-symlinks False --local-dir /PATH/TO/VILA_MODEL
```

你可以选择性地准备 LLMs 来改写上述视频描述的结果。例如，你执行下面的命令来下载 [Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct)
```shell
# Add HF_ENDPOINT=https://hf-mirror.com before the command if you cannot access to huggingface.com
huggingface-cli download NousResearch/Meta-Llama-3-8B-Instruct --local-dir-use-symlinks False --local-dir /PATH/TO/REWRITE_MODEL
```

视频描述的完整流程在 [stage_3_video_recaptioning.sh](./scripts/stage_3_video_recaptioning.sh).
执行
```shell
VILA_MODEL_PATH=/PATH/TO/VILA_MODEL REWRITE_MODEL_PATH=/PATH/TO/REWRITE_MODEL sh scripts/stage_3_video_recaptioning.sh
```
后，最后的训练文件会保存在 `easyanimate/video_caption/datasets/panda_70m/videos_clips/meta_train_info.json`。