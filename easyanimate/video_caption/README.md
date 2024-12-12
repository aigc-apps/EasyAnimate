# Video Caption
English | [简体中文](./README_zh-CN.md)

The folder contains codes for dataset preprocessing (i.e., video splitting, filtering, and recaptioning), and beautiful prompt used by EasyAnimate.
The entire process supports distributed parallel processing, capable of handling large-scale datasets.

Meanwhile, we are collaborating with [Data-Juicer](https://github.com/modelscope/data-juicer/blob/main/docs/DJ_SORA.md),
allowing you to easily perform video data processing on [Aliyun PAI-DLC](https://help.aliyun.com/zh/pai/user-guide/video-preprocessing/).

# Table of Content
- [Video Caption](#video-caption)
- [Table of Content](#table-of-content)
  - [Quick Start](#quick-start)
    - [Setup](#setup)
    - [Data Preprocessing](#data-preprocessing)
      - [Data Preparation](#data-preparation)
      - [Video Splitting](#video-splitting)
      - [Video Filtering](#video-filtering)
      - [Video Recaptioning](#video-recaptioning)
    - [Beautiful Prompt (For EasyAnimate Inference)](#beautiful-prompt-for-easyanimate-inference)
      - [Batched Inference](#batched-inference)
      - [OpenAI Server](#openai-server)

## Quick Start

### Setup
AliyunDSW or Docker is recommended to setup the environment, please refer to [Quick Start](../../README.md#quick-start).
You can also refer to the image build process in the [Dockerfile](../../Dockerfile.ds) to configure the conda environment and other dependencies locally.

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

### Data Preprocessing
#### Data Preparation
Place the downloaded videos into a folder under [datasets](./datasets/) (preferably without nested structures, as the video names are used as unique IDs in subsequent processes).
Taking Panda-70M as an example, the entire dataset directory structure is shown as follows:
```
📦 datasets/
├── 📂 panda_70m/
│   ├── 📂 videos/
│   │   ├── 📂 data/
│   │   │   └── 📄 --C66yU3LjM_2.mp4
│   │   │   └── 📄 ...
```

#### Video Splitting
EasyAnimate utilizes [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) to identify scene changes within the video
and performs video splitting via FFmpeg based on certain threshold values to ensure consistency of the video clip.
Video clips shorter than 3 seconds will be discarded, and those longer than 10 seconds will be splitted recursively.

The entire workflow of video splitting is in the [stage_1_video_splitting.sh](./scripts/stage_1_video_splitting.sh).
After running
```shell
sh scripts/stage_1_video_splitting.sh
```
the video clips are obtained in `easyanimate/video_caption/datasets/panda_70m/videos_clips/data/`.

#### Video Filtering
Based on the videos obtained in the previous step, EasyAnimate provides a simple yet effective pipeline to filter out high-quality videos for recaptioning.
The overall process is as follows:

- Scene transition filtering: Filter out videos with scene transition introduced by missing or superfluous splitting of PySceneDetect by calculating the semantic similarity
accoss the beginning frame, the last frame, and the keyframes via [CLIP](https://github.com/openai/CLIP) or [DINOv2](https://github.com/facebookresearch/dinov2).
- Aesthetic filtering: Filter out videos with poor content (blurry, dim, etc.) by calculating the average aesthetic score of uniformly sampled 4 frames via [aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5).
- Text filtering: Use [EasyOCR](https://github.com/JaidedAI/EasyOCR) to calculate the text area proportion of the middle frame to filter out videos with a large area of text.
- Motion filtering: Calculate interframe optical flow differences to filter out videos that move too slowly or too quickly.

The entire workflow of video filtering is in the [stage_2_video_filtering.sh](./scripts/stage_2_video_filtering.sh).
After running
```shell
sh scripts/stage_2_video_filtering.sh
```
the semantic consistency score, aesthetic score, text score, and motion score of videos will be saved 
in the corresponding meta files in the folder `easyanimate/video_caption/datasets/panda_70m/videos_clips/`.

> [!NOTE]
> The computation of semantic consistency score depends on the [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336).
Meanwhile, the aesthetic score depends on the [google/siglip-so400m-patch14-384 model](https://huggingface.co/google/siglip-so400m-patch14-384).
Please run `HF_ENDPOINT=https://hf-mirror.com sh scripts/stage_2_video_filtering.sh` if you cannot access to huggingface.com.


#### Video Recaptioning
After obtaining the aboved high-quality filtered videos, EasyAnimate utilizes [InternVL2](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html) to perform video recaptioning.
Subsequently, the recaptioning results are rewritten by LLMs to better meet with the requirements of video generation tasks.
Finally, an advanced [VideoCLIP-XL](https://arxiv.org/abs/2410.00741) model is used to filter out (video, long caption) pairs with poor alignment, resulting in the final training dataset.

Please download the video caption model from [InternVL2](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e) of the appropriate size based on the GPU memory of your machine.
For A100 with 40G VRAM, you can download [InternVL2-40B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-40B-AWQ) by running
```shell
# Add HF_ENDPOINT=https://hf-mirror.com before the command if you cannot access to huggingface.com
huggingface-cli download OpenGVLab/InternVL2-40B-AWQ --local-dir-use-symlinks False --local-dir /PATH/TO/INTERNVL2_MODEL
```

Optionally, you can prepare local LLMs to rewrite the recaption results.
For example, you can download [Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct) by running
```shell
# Add HF_ENDPOINT=https://hf-mirror.com before the command if you cannot access to huggingface.com
huggingface-cli download NousResearch/Meta-Llama-3-8B-Instruct --local-dir-use-symlinks False --local-dir /PATH/TO/REWRITE_MODEL
```

The entire workflow of video recaption is in the [stage_3_video_recaptioning.sh](./scripts/stage_3_video_recaptioning.sh).
After running
```shell
CAPTION_MODEL_PATH=/PATH/TO/INTERNVL2_MODEL REWRITE_MODEL_PATH=/PATH/TO/REWRITE_MODEL sh scripts/stage_3_video_recaptioning.sh
``` 
the final train file is obtained in `easyanimate/video_caption/datasets/panda_70m/videos_clips/meta_train_info.json`.


### Beautiful Prompt (For EasyAnimate Inference)
Beautiful Prompt aims to rewrite and beautify the user-uploaded prompt via LLMs, mapping it to the style of EasyAnimate's training captions,
making it more suitable as the inference prompt and thus improving the quality of the generated videos.
We support batched inference with local LLMs or OpenAI compatible server based on [vLLM](https://github.com/vllm-project/vllm) for beautiful prompt.

#### Batched Inference
1. Prepare original prompts in a jsonl file `easyanimate/video_caption/datasets/original_prompt.jsonl` with the following format:
    ```json
    {"prompt": "A stylish woman in a black leather jacket, red dress, and boots walks confidently down a damp Tokyo street."}
    {"prompt": "An underwater world with realistic fish and other creatures of the sea."}
    {"prompt": "a monarch butterfly perched on a tree trunk in the forest."}
    {"prompt": "a child in a room with a bottle of wine and a lamp."}
    {"prompt": "two men in suits walking down a hallway."}
    ```

2. Then you can perform beautiful prompt by running
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

#### OpenAI Server
+ You can request OpenAI compatible server to perform beautiful prompt by running
    ```shell
    OPENAI_API_KEY="your_openai_api_key" OPENAI_BASE_URL="your_openai_base_url" python beautiful_prompt.py \
        --model "your_model_name" \
        --prompt "your_prompt"
    ```

+ You can also deploy the OpenAI Compatible Server locally using vLLM. For example:
    ```shell
    # Meta-Llama-3-8B-Instruct is sufficient for this task.
    # Download it from https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct or https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct to /path/to/your_llm

    # deploy the OpenAI compatible server
    python -m vllm.entrypoints.openai.api_server serve /path/to/your_llm --dtype auto --api-key "your_api_key"
    ```

    Then you can perform beautiful prompt by running
    ```shell
    python -m beautiful_prompt.py \
        --model /path/to/your_llm \
        --prompt "your_prompt" \
        --base_url "http://localhost:8000/v1" \
        --api_key "your_api_key"
    ```
