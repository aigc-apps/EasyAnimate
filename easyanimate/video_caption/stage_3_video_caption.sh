export CUDA_VISIBLE_DEVICES=0
export VIDEO_FOLDER="datasets/panda_70m/train/"
export MOTION_SCORE_META_PATH="datasets/panda_70m/train.jsonl"
export VIDEO_FRAME_CAPTION_PATH="datasets/panda_70m/frame_caption.jsonl"
export VIDEO_CAPTION_PATH="datasets/panda_70m/summary_caption.jsonl"
export LAST_JSON_PATH="datasets/panda_70m/train_panda_70m.json"

CUDA_VISIBLE_DEVICES="0" python caption_video_frame.py \
    --video_metadata_path=$MOTION_SCORE_META_PATH \
    --video_folder=$VIDEO_FOLDER \
    --frame_sample_method="mid" \
    --num_sampled_frames=1 \
    --image_caption_model_name="llava-v1.6-vicuna-7b" \
    --image_caption_prompt="Please describe this image in detail." \
    --saved_path=$VIDEO_FRAME_CAPTION_PATH \
    --output_dir="tmp"

CUDA_VISIBLE_DEVICES="0" python caption_summary.py \
    --video_metadata_path=$VIDEO_FRAME_CAPTION_PATH \
    --video_path_column="video_path" \
    --caption_column="sampled_frame_caption" \
    --summary_model_name="Qwen/Qwen1.5-7B-Chat" \
    --summary_prompt="You are a helpful video description generator. I'll give you a description of the middle frame of the video clip,  \
    which you need to summarize it into a description of the video clip. \
    Please provide your video description following these requirements: \
    1. Describe the basic and necessary information of the video in the third person, be as concise as possible. \
    2. Output the video description directly. Begin with 'In this video'. \
    3. Limit the video description within 100 words. \
    Here is the mid-frame description: " \
    --saved_path=$VIDEO_CAPTION_PATH

python convert_jsonl_to_json.py \
    --video_folder=$VIDEO_FOLDER \
    --jsonl_load_path=$VIDEO_CAPTION_PATH \
    --save_path=$LAST_JSON_PATH