# Video Caption
EasyAnimate uses multi-modal LLMs to generate captions for frames extracted from the video firstly, and then employs LLMs to summarize and refine the generated frame captions into the final video caption. By leveraging [sglang](https://github.com/sgl-project/sglang)/[vLLM](https://github.com/vllm-project/vllm) and [accelerate distributed inference](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference), the entire processing could be very fast.

English | [简体中文](./README_zh-CN.md)

## Quick Start
1. Cloud usage: AliyunDSW/Docker
    
    Check [README.md](../../README.md#quick-start) for details.

2. Local usage

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

## Data preprocessing
Data preprocessing can be divided into three parts:

- Video cut.
- Video cleaning.
- Video caption.

The input for data preprocessing can be a video folder or a metadata file (txt/csv/jsonl) containing the video path column. Please check `get_video_path_list` function in [utils/video_utils.py](utils/video_utils.py) for details.

For easier understanding, we use one data from Panda70m as an example for data preprocessing, [Download here](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v2/--C66yU3LjM_2.mp4). Please download the video and push it in "datasets/panda_70m/before_vcut/"

```
📦 datasets/
├── 📂 panda_70m/
│   └── 📂 before_vcut/
│       └── 📄 --C66yU3LjM_2.mp4
```

1. Video cut

    For long video cut, EasyAnimate utilizes PySceneDetect to identify scene changes within the video and performs scene cutting based on certain threshold values to ensure consistency in the themes of the video segments. After cutting, we only keep segments with lengths ranging from 3 to 10 seconds for model training.

    We have completed the parameters for ```stage_1_video_cut.sh```, so I can run it directly using the command sh ```stage_1_video_cut.sh```. After executing ```stage_1_video_cut.sh```, we obtained short videos in ```easyanimate/video_caption/datasets/panda_70m/train```.

    ```shell
    sh stage_1_video_cut.sh
    ```
2. Video cleaning

    Following SVD's data preparation process, EasyAnimate provides a simple yet effective data processing pipeline for high-quality data filtering and labeling. It also supports distributed processing to accelerate the speed of data preprocessing. The overall process is as follows:

   - Duration filtering: Analyze the basic information of the video to filter out low-quality videos that are short in duration or low in resolution. This filtering result is corresponding to the video cut (3s ~ 10s videos).
   - Aesthetic filtering: Filter out videos with poor content (blurry, dim, etc.) by calculating the average aesthetic score of uniformly distributed 4 frames.
   - Text filtering: Use easyocr to calculate the text proportion of middle frames to filter out videos with a large proportion of text.
   - Motion filtering: Calculate interframe optical flow differences to filter out videos that move too slowly or too quickly.

    The process file of **Aesthetic filtering** is ```compute_video_frame_quality.py```. After executing ```compute_video_frame_quality.py```, we obtained the file ```datasets/panda_70m/aesthetic_score.jsonl```, where each line corresponds to the aesthetic score of each video.

    The process file of **Text filtering** is ```compute_text_score.py```. After executing ```compute_text_score.py```, we obtained the file ```datasets/panda_70m/text_score.jsonl```, where each line corresponds to the text score of each video.

    The process file of **Motion filtering** is ```compute_motion_score.py```. Motion filtering is based on Aesthetic filtering and Text filtering; only samples that meet certain aesthetic scores and text scores will undergo calculation for the Motion score. After executing ```compute_motion_score.py```, we obtained the file ```datasets/panda_70m/motion_score.jsonl```, where each line corresponds to the motion score of each video.

    Then we need to filter videos by motion scores. After executing ```filter_videos_by_motion_score.py```, we get the file ```datasets/panda_70m/train.jsonl```, which includes the video we need to caption.

    We have completed the parameters for stage_2_filter_data.sh, so I can run it directly using the command sh stage_2_filter_data.sh.

    ```shell
    sh stage_2_filter_data.sh
    ```
3. Video caption

    Video captioning is carried out in two stages. The first stage involves extracting frames from a video and generating descriptions for them. Subsequently, a large language model is used to summarize these descriptions into a caption.

    We have conducted a detailed and manual comparison of open sourced multi-modal LLMs such as [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL), [ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B), [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat) and etc. And we found that [llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) is capable of generating more detailed captions with fewer hallucinations. Additionally, it is supported by serving engines like [sglang](https://github.com/sgl-project/sglang) and [lmdepoly](https://github.com/InternLM/lmdeploy), enabling faster inference.

    Firstly, we use ```caption_video_frame.py``` to generate frame captions. Then, we use ```caption_summary.py``` to generate summary captions.

    We have completed the parameters for stage_3_video_caption.sh, so I can run it directly using the command sh stage_3_video_caption.sh. After executing ```stage_3_video_cut.sh```, we obtained last json ```train_panda_70m.json``` for easyanimate training. 

    ```shell
    sh stage_3_video_caption.sh
    ```

    If you cannot access to Huggingface, you can run `export HF_ENDPOINT=https://hf-mirror.com` before the above command to download the summary caption model automatically.