META_FILE_PATH="datasets/panda_70m/videos_clips/data/meta_file_info.jsonl"
VIDEO_FOLDER="datasets/panda_70m/videos_clips/data/"
MOTION_SCORE_SAVED_PATH="datasets/panda_70m/videos_clips/meta_motion_info.jsonl"
MIN_MOTION_SCORE=2
VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b.jsonl"
REWRITTEN_VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b_rewritten.jsonl"
LAST_JSON_PATH="datasets/panda_70m/train_panda_70m.json"
# Manually download Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ to VILA_MODEL_PATH.
VILA_MODEL_PATH="/PATH/TO/Llama-3-VILA1.5-8b-AWQ"
# Manually download meta-llama/Meta-Llama-3-8B-Instruct to REWRITE_MODEL_PATH.
REWRITE_MODEL_PATH="/PATH/TO/Meta-Llama-3-8B-Instruct"

# Use Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ to perform recaption.
accelerate launch vila_caption_video.py \
    --video_metadata_path META_FILE_PATH \
    --video_folder ${VIDEO_FOLDER} \
    --model_path ${MODEL_PATH} \
    --precision "W4A16" \
    --saved_path $VIDEO_CAPTION_SAVED_PATH \
    --saved_freq 1 \
    --motion_score_metadata_path $MOTION_SCORE_SAVED_PATH \
    --min_motion_score $MIN_MOTION_SCORE

# Rewrite video captions (optional)
python caption_rewrite.py \
    --video_metadata_path $VIDEO_CAPTION_SAVED_PATH \
    --batch_size 4096 \
    --model_name $REWRITE_MODEL_PATH \
    --prompt prompt/rewrite.txt \
    --saved_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --saved_freq 1

python -m utils.convert_jsonl_to_json \
    --video_folder=$VIDEO_FOLDER \
    --jsonl_load_path=$REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --save_path=$LAST_JSON_PATH