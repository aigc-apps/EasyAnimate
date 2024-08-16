# Video Caption
English | [简体中文](./README_zh-CN.md)

The folder contains codes for dataset preprocessing (i.e., video splitting), filtering, and recaptioning used by EasyAnimate.

## Quick Start

### Setup
AliyunDSW or Docker is recommended to setup the environment, please refer to [Quick Start](../../README.md#quick-start).
You can also refer to the image build process in the [Dockerfile](../../Dockerfile.ds) to configure the conda environment and other dependencies locally.

Since the video recaptioning depends on [llm-awq](https://github.com/mit-han-lab/llm-awq) for faster and memory efficient inference,
the minimum GPU requirment should be RTX 3060 or A2 (CUDA Compute Capability >= 8.0).

```shell
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:
asyanimate_video_caption

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:asyanimate_video_caption

# clone code
git clone https://github.com/aigc-apps/EasyAnimate.git

# enter video_caption
cd EasyAnimate/easyanimate/video_caption
```

### Data Preparation
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

### Video Splitting
EasyAnimate utilizes [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) to identify scene changes within the video
and performs video splitting via FFmpeg based on certain threshold values to ensure consistency of the video clip.
Video clips shorter than 3 seconds will be discarded, and those longer than 10 seconds will be splitted recursively.

The entire workflow of video splitting is in the [stage_1_video_splitting.sh](./scripts/stage_1_video_splitting.sh).
After running
```shell
sh scripts/stage_1_video_splitting.sh
```
the video clips are obtained in `easyanimate/video_caption/datasets/panda_70m/videos_clips/data/`.

### Video Filtering
Based on the videos obtained in the previous step, EasyAnimate provides a simple yet effective pipeline to filter out high-quality videos for recaptioning.
The overall process is as follows:

- Aesthetic filtering: Filter out videos with poor content (blurry, dim, etc.) by calculating the average aesthetic score of uniformly sampled 4 frames via [aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5).
- Text filtering: Use [EasyOCR](https://github.com/JaidedAI/EasyOCR) to calculate the text area proportion of the middle frame to filter out videos with a large area of text.
- Motion filtering: Calculate interframe optical flow differences to filter out videos that move too slowly or too quickly.

The entire workflow of video filtering is in the [stage_2_video_filtering.sh](./scripts/stage_2_video_filtering.sh).
After running
```shell
sh scripts/stage_2_video_filtering.sh
```
the aesthetic score, text score, and motion score of videos will be saved in the corresponding meta files in the folder `easyanimate/video_caption/datasets/panda_70m/videos_clips/`.

> [!NOTE]
> The computation of the aesthetic score depends on the [google/siglip-so400m-patch14-384 model](https://huggingface.co/google/siglip-so400m-patch14-384).
Please run `HF_ENDPOINT=https://hf-mirror.com sh scripts/stage_2_video_filtering.sh` if you cannot access to huggingface.com.


### Video Recaptioning
After obtaining the aboved high-quality filtered videos, EasyAnimate utilizes [VILA1.5](https://github.com/NVlabs/VILA) to perform video recaptioning. 
Subsequently, the recaptioning results are rewritten by LLMs to better meet with the requirements of video generation tasks. 
Finally, an advanced VideoCLIPXL model is developed to filter out video-caption pairs with poor alignment, resulting in the final training dataset.

Please download the video caption model from [VILA1.5](https://huggingface.co/collections/Efficient-Large-Model/vila-on-pre-training-for-visual-language-models-65d8022a3a52cd9bcd62698e) of the appropriate size based on the GPU memory of your machine.
For A100 with 40G VRAM, you can download [VILA1.5-40b-AWQ](https://huggingface.co/Efficient-Large-Model/VILA1.5-40b-AWQ) by running
```shell
# Add HF_ENDPOINT=https://hf-mirror.com before the command if you cannot access to huggingface.com
huggingface-cli download Efficient-Large-Model/VILA1.5-40b-AWQ --local-dir-use-symlinks False --local-dir /PATH/TO/VILA_MODEL
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
VILA_MODEL_PATH=/PATH/TO/VILA_MODEL REWRITE_MODEL_PATH=/PATH/TO/REWRITE_MODEL sh scripts/stage_3_video_recaptioning.sh
``` 
the final train file is obtained in `easyanimate/video_caption/datasets/panda_70m/videos_clips/meta_train_info.json`.