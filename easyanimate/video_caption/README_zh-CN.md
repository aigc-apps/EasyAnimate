# 数据预处理


EasyAnimate 对数据进行了场景切分、视频过滤和视频打标来得到高质量的有标注视频训练使用。使用多模态大型语言模型(LLMs)为从视频中提取的帧生成字幕，然后利用LLMs将生成的帧字幕总结并细化为最终的视频字幕。通过利用sglang/vLLM和加速分布式推理，高效完成视频的打标。

[English](./README.md) | 简体中文

## 快速开始
1. 云上使用: 阿里云DSW/Docker
    参考 [README.md](../../README_zh-CN.md#quick-start) 查看更多细节。

2. 本地安装

    ```shell
    # Install EasyAnimate requirements firstly.
    cd EasyAnimate && pip install -r requirements.txt

    # Install additional requirements for video caption.
    cd easyanimate/video_caption && pip install -r requirements.txt --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

    # Use DDP instead of DP in EasyOCR detection.
    site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
    cp -v easyocr_detection_patched.py $site_pkg_path/easyocr/detection.py

    # We strongly recommend using Docker unless you can properly handle the dependency between vllm with torch(cuda).
    ```

## 数据预处理
数据预处理可以分为一下三步：

- 视频切分
- 视频过滤
- 视频打标

数据预处理的输入可以是视频文件夹或包含视频路径列的元数据文件（txt/csv/jsonl格式）。详情请查看[utils/video_utils.py](utils/video_utils.py) 文件中的 `get_video_path_list` 函数。

为了便于理解，我们以Panda70m的一个数据为例进行数据预处理，点击[这里](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v2/--C66yU3LjM_2.mp4)下载视频。请下载视频并放在下面的路径："datasets/panda_70m/before_vcut/"

```
📦 datasets/
├── 📂 panda_70m/
│   └── 📂 before_vcut/
│       └── 📄 --C66yU3LjM_2.mp4
```

1. 视频切分

    对于长视频剪辑，EasyAnimate 利用 PySceneDetect 来识别视频中的场景变化，并根据特定的阈值进行场景切割，以确保视频片段主题的一致性。切割后，我们只保留长度在3到10秒之间的片段，用于模型训练。

    我们整理了完整的方案在 ```stage_1_video_cut.sh``` 文件中, 您可以直接运行```stage_1_video_cut.sh```. 执行完成后可以在 ```easyanimate/video_caption/datasets/panda_70m/train``` 文件夹中查看结果。

    ```shell
    sh stage_1_video_cut.sh
    ```
2. 视频过滤

    遵循SVD([Stable Video Diffusion](https://github.com/Stability-AI/generative-models))的数据准备流程，EasyAnimate 提供了一个简单而有效的数据处理管道，用于高质量数据的过滤和标记。我们还支持分布式处理来加快数据预处理的速度。整个过程如下：:

   - 时长过滤: 分析视频的基本信息，筛选出时长过短或分辨率过低的低质量视频。我们保留3秒至10秒的视频。
   - 美学过滤: 通过计算均匀分布的4帧的平均审美分数，过滤掉内容质量差的视频（模糊、暗淡等）。
   - 文本过滤: 使用 [easyocr](https://github.com/JaidedAI/EasyOCR) 来计算中间帧的文本比例，以筛选出含有大量文本的视频。
   - 运动过滤: 计算帧间光流差异，以筛选出移动过慢或过快的视频。

    **美学过滤** 的代码在 ```compute_video_frame_quality.py```. 执行 ```compute_video_frame_quality.py```,我们可以生成 ```datasets/panda_70m/aesthetic_score.jsonl```文件, 计算每条视频的美学得分。

    **文本过滤** 的代码在 ```compute_text_score.py```. 执行```compute_text_score.py```, 我们可以生成 ```datasets/panda_70m/text_score.jsonl```文件, 计算每个视频的文字占比。

    **运动过滤** 的代码在 ```compute_motion_score.py```. 运动过滤基于审美过滤和文本过滤；只有达到一定审美分数和文本分数的样本才会进行运动分数的计算。 执行 ```compute_motion_score.py```, 我们可以生成 ```datasets/panda_70m/motion_score.jsonl```, 计算每条视频的运动得分。

    接着执行 ```filter_videos_by_motion_score.py```来得过滤视频。我们最终得到筛选后需要打标的 ```datasets/panda_70m/train.jsonl```文件。

    我们将视频过滤的流程整理为 ```stage_2_filter_data.sh```，直接执行该脚本来完成视频数据的过滤。

    ```shell
    sh stage_2_filter_data.sh
    ```
3. 视频打标

    
    视频打标生成分为两个阶段。第一阶段涉及从视频中提取帧并为它们生成描述。随后，使用大型语言模型将这些描述汇总成一条字幕。

    我们详细对比了现有的多模态大语言模型（诸如[Qwen-VL](https://huggingface.co/Qwen/Qwen-VL), [ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B), [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)）生成文本描述的效果。 最终选择 [llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) 来进行视频文本描述的生成，它能生成详细的描述并有更少的幻觉。此外，我们引入 [sglang](https://github.com/sgl-project/sglang)，[lmdepoly](https://github.com/InternLM/lmdeploy), 来加速推理的过程。

    首先，我们用 ```caption_video_frame.py``` 来生成文本描述，并用 ```caption_summary.py``` 来总结描述信息。我们将上述过程整理在 ```stage_3_video_caption.sh```, 直接运行它来生成视频的文本描述。我们最终得到 ```train_panda_70m.json``` 用于EasyAnmate 的训练。 

    ```shell
    sh stage_3_video_caption.sh
    ```

   请注意，如遇网络问题，您可以设置 `export HF_ENDPOINT=https://hf-mirror.com` 来自动下载视频打标模型。