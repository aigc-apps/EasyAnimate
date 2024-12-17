# 数据预处理
[English](./README.md) | 简体中文

该文件夹包含 EasyAnimate 使用的数据集预处理（即视频切分、过滤和生成描述）和提示词美化的代码。整个过程支持分布式并行处理，能够处理大规模数据集。

此外，我们和 [Data-Juicer](https://github.com/modelscope/data-juicer/blob/main/docs/DJ_SORA.md) 合作，能让你在 [Aliyun PAI-DLC](https://help.aliyun.com/zh/pai/user-guide/video-preprocessing/) 轻松进行视频数据的处理。

# 目录
- [数据预处理](#数据预处理)
- [目录](#目录)
  - [快速开始](#快速开始)
    - [安装](#安装)
    - [数据集预处理](#数据集预处理)
      - [数据准备](#数据准备)
      - [视频切分](#视频切分)
      - [视频过滤](#视频过滤)
      - [视频描述](#视频描述)
    - [提示词美化](#提示词美化)
      - [批量推理](#批量推理)
      - [OpenAI 服务器](#openai-服务器)


## 快速开始
### 安装
推荐使用阿里云 DSW 和 Docker 来安装环境，请参考 [快速开始](../../README_zh-CN.md#1-云使用-aliyundswdocker). 你也可以参考 [Dockerfile](../../Dockerfile.ds) 中的镜像构建流程在本地安装对应的 conda 环境和其余依赖。

```shell
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# clone code
git clone https://github.com/aigc-apps/EasyAnimate.git

# enter video_caption
cd EasyAnimate/easyanimate/video_caption
```

### 数据集预处理
#### 数据准备
将下载的视频准备到文件夹 [datasets](./datasets/)（最好不使用嵌套结构，因为视频名称在后续处理中用作唯一 ID）。以 Panda-70M 为例，完整的数据集目录结构如下所示：
```
📦 datasets/
├── 📂 panda_70m/
│   ├── 📂 videos/
│   │   ├── 📂 data/
│   │   │   └── 📄 --C66yU3LjM_2.mp4
│   │   │   └── 📄 ...
```

#### 视频切分
EasyAnimate 使用 [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) 来识别视频中的场景变化
并根据某些阈值通过 FFmpeg 执行视频分割，以确保视频片段的一致性。
短于 3 秒的视频片段将被丢弃，长于 10 秒的视频片段将被递归切分。

视频切分的完整流程在 [stage_1_video_splitting.sh](./scripts/stage_1_video_splitting.sh)。执行
```shell
sh scripts/stage_1_video_splitting.sh
```
后，切分后的视频位于 `easyanimate/video_caption/datasets/panda_70m/videos_clips/data/`。

#### 视频过滤
基于上一步获得的视频，EasyAnimate 提供了一个简单而有效的流程来过滤出高质量的视频。总体流程如下：

- 场景跳变过滤：通过 [CLIP](https://github.com/openai/CLIP) 或者 [DINOv2](https://github.com/facebookresearch/dinov2) 来计算关键帧和首尾帧的语义相似度，从而过滤掉由于 PySceneDetect 缺失或多余分割引入的场景跳变的视频。
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

#### 视频描述
在获得上述高质量的过滤视频后，EasyAnimate 利用 [InternVL2](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html) 来生成视频描述。随后，使用 LLMs 对生成的视频描述进行重写，以更好地满足视频生成任务的要求。最后，使用自研的 [VideoCLIP-XL](https://arxiv.org/abs/2410.00741) 模型来过滤掉描述和视频内容不一致的数据，从而得到最终的训练数据集。

请根据机器的显存从 [InternVL2](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e) 下载合适大小的模型。对于 A100 40G，你可以执行下面的命令来下载 [InternVL2-40B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-40B-AWQ)
```shell
# Add HF_ENDPOINT=https://hf-mirror.com before the command if you cannot access to huggingface.com
huggingface-cli download OpenGVLab/InternVL2-40B-AWQ --local-dir-use-symlinks False --local-dir /PATH/TO/INTERNVL2_MODEL
```

你可以选择性地准备 LLMs 来改写上述视频描述的结果。例如，你执行下面的命令来下载 [Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct)
```shell
# Add HF_ENDPOINT=https://hf-mirror.com before the command if you cannot access to huggingface.com
huggingface-cli download NousResearch/Meta-Llama-3-8B-Instruct --local-dir-use-symlinks False --local-dir /PATH/TO/REWRITE_MODEL
```

视频描述的完整流程在 [stage_3_video_recaptioning.sh](./scripts/stage_3_video_recaptioning.sh).
执行
```shell
CAPTION_MODEL_PATH=/PATH/TO/INTERNVL2_MODEL REWRITE_MODEL_PATH=/PATH/TO/REWRITE_MODEL sh scripts/stage_3_video_recaptioning.sh
```
后，最后的训练文件会保存在 `easyanimate/video_caption/datasets/panda_70m/videos_clips/meta_train_info.json`。

### 提示词美化
提示词美化旨在通过 LLMs 重写和美化用户上传的提示，将其映射为 EasyAnimate 训练所使用的视频描述风格、
使其更适合用作推理提示词，从而提高生成视频的质量。

基于 [vLLM](https://github.com/vllm-project/vllm)，我们支持使用本地 LLM 进行批量推理或请求 OpenAI 服务器的方式，以进行提示词美化。

#### 批量推理
1. 将原始的提示词以下面的格式准备在文件 `easyanimate/video_caption/datasets/original_prompt.jsonl` 中：
    ```json
    {"prompt": "A stylish woman in a black leather jacket, red dress, and boots walks confidently down a damp Tokyo street."}
    {"prompt": "An underwater world with realistic fish and other creatures of the sea."}
    {"prompt": "a monarch butterfly perched on a tree trunk in the forest."}
    {"prompt": "a child in a room with a bottle of wine and a lamp."}
    {"prompt": "two men in suits walking down a hallway."}
    ```

2. 随后你可以通过执行以下的命令进行提示词美化
    ```shell
    # Meta-Llama-3-8B-Instruct is sufficient for this task.
    # Download it from https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct or https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct to /path/to/your_llm

    python caption_rewrite.py \
        --video_metadata_path datasets/original_prompt.jsonl \
        --caption_column "prompt" \
        --beautiful_prompt_column "beautiful_prompt" \
        --batch_size 1 \
        --model_name /path/to/your_llm \
        --prompt prompt/beautiful_prompt.txt \
        --prefix '"detailed description": ' \
        --answer_template "your detailed description here" \
        --max_retry_count 10 \
        --saved_path datasets/beautiful_prompt.jsonl \
        --saved_freq 1
    ```

#### OpenAI 服务器
+ 你可以通过请求 OpenAI 服务器的方式来进行提示词美化
    ```shell
    OPENAI_API_KEY="your_openai_api_key" OPENAI_BASE_URL="your_openai_base_url" python beautiful_prompt.py \
        --model "your_model_name" \
        --prompt "your_prompt"
    ```

+ 你也可以执行以下命令，通过 vLLM 将本地 LLMs 部署成兼容 OpenAI 的服务器
    ```shell
    OPENAI_API_KEY="your_openai_api_key" OPENAI_BASE_URL="your_openai_base_url" python beautiful_prompt.py \
        --model "your_model_name" \
        --prompt "your_prompt"
    ```

    然后再执行下面的命令来进行提示词美化
    ```shell
    python -m beautiful_prompt.py \
        --model /path/to/your_llm \
        --prompt "your_prompt" \
        --base_url "http://localhost:8000/v1" \
        --api_key "your_api_key"
    ```