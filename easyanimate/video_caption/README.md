# Video Caption
EasyAnimate uses multi-modal LLMs to generate captions for frames extracted from the video firstly, and then employs LLMs to summarize and refine the generated frame captions into the final video caption. By leveraging [sglang](https://github.com/sgl-project/sglang)/[vLLM](https://github.com/vllm-project/vllm) and [accelerate distributed inference](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference), the entire processing could be very fast.

## Quick Start
1. Cloud usage: AliyunDSW/Docker
    
    Check [README.md](../../README.md#quick-start) for details.

2. Local usage

    ```shell
    # Install EasyAnimate requirements firstly.
    cd EasyAnimate && pip install -r requirements.txt

    # Install additional requirements for video caption.
    cd easyanimate/video_caption && pip install -r requirements.txt

    # We strongly recommend using Docker unless you can properly handle the dependency between vllm with torch(cuda).
    ```

## How to use

1. Prepare videos.

    The input for video caption can be a video folder or a metadata file (txt/csv/jsonl) containing the video path column. Please check `get_video_path_list` function in [utils/video_utils.py](utils/video_utils.py) for details.

2. Generate frame captions.

    We have conducted a detailed and manual comparison of open sourced multi-modal LLMs such as [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL), [ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B), [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat) and etc. And we found that [llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) is capable of generating more detailed captions with fewer hallucinations. Additionally, it is supported by serving engines like [sglang](https://github.com/sgl-project/sglang) and [lmdepoly](https://github.com/InternLM/lmdeploy), enabling faster inference.

    ```shell
    CUDA_VISIBLE_DEVICES=0 python caption_video_frame.py \
        --video_folder="your-video-folder/"
        --frame_sample_method="mid" \
        --num_sampled_frames=1 \
        --image_caption_model_name="llava-v1.6-vicuna-7b" \
        --image_caption_prompt="Please describe this image in detail." \
        --saved_path="video_frame_caption.jsonl"
    ```

    If you cannot access to Huggingface, you can run `export HF_ENDPOINT=https://hf-mirror.com` before the above command to download the image caption model automatically.

3. Summary frame captions.

    ```shell
    CUDA_VISIBLE_DEVICES=0 python caption_summary.py \
        --video_metadata_path="video_frame_caption_result.jsonl" \
        --video_path_column="video_path" \
        --caption_column="sampled_frame_caption" \
        --summary_model_name="mistralai/Mistral-7B-Instruct-v0.2" \
        --summary_prompt="You are a helpful video description generator. I'll give you a description of the middle frame of the video clip,  \
        which you need to summarize it into a description of the video clip. \
        Please provide your video description following these requirements: \
        1. Describe the basic and necessary information of the video in the third person, be as concise as possible. \
        2. Output the video description directly. Begin with 'In this video'. \
        3. Limit the video description within 100 words. \
        Here is the mid-frame description: " \
        --output_dir="tmp" \
        --saved_path="video_summary_caption.jsonl"
    ```

    If you cannot access to Huggingface, you can run `export HF_ENDPOINT=https://hf-mirror.com` before the above command to download the summary caption model automatically.