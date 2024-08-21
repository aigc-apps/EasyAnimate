META_FILE_PATH="datasets/panda_70m/videos_clips/data/meta_file_info.jsonl"
VIDEO_FOLDER="datasets/panda_70m/videos_clips/data/"
MOTION_SAVED_PATH="datasets/panda_70m/videos_clips/meta_motion_info.jsonl"
MIN_MOTION_SCORE=2
VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b.jsonl"
REWRITTEN_VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b_rewritten.jsonl"
VIDEOCLIPXL_SCORE_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b_rewritten_videoclipxl.jsonl"
MIN_VIDEOCLIPXL_SCORE=0.20
TRAIN_SAVED_PATH="datasets/panda_70m/train_panda_70m.json"
# Manually download Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ to VILA_MODEL_PATH.
# Manually download meta-llama/Meta-Llama-3-8B-Instruct to REWRITE_MODEL_PATH.

# Use VILA1.5-AWQ to perform recaptioning.
accelerate launch vila_video_recaptioning.py \
    --video_metadata_path ${META_FILE_PATH} \
    --video_folder ${VIDEO_FOLDER} \
    --model_path ${VILA_MODEL_PATH} \
    --precision "W4A16" \
    --saved_path $VIDEO_CAPTION_SAVED_PATH \
    --saved_freq 1 \
    --motion_score_metadata_path $MOTION_SAVED_PATH \
    --min_motion_score $MIN_MOTION_SCORE

# Rewrite video captions (optional).
python caption_rewrite.py \
    --video_metadata_path $VIDEO_CAPTION_SAVED_PATH \
    --batch_size 4096 \
    --model_name $REWRITE_MODEL_PATH \
    --prompt prompt/rewrite.txt \
    --prefix '"rewritten description": ' \
    --saved_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --saved_freq 1

# Compute caption-video alignment (optional).
accelerate launch compute_video_quality.py \
    --video_metadata_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --caption_column caption \
    --video_folder $VIDEO_FOLDER \
    --frame_sample_method uniform \
    --num_sampled_frames 8 \
    --metrics VideoCLIPXLScore \
    --batch_size 4 \
    --saved_path $VIDEOCLIPXL_SCORE_SAVED_PATH \
    --saved_freq 10

# Get the final train file.
python filter_meta_train.py \
    --caption_metadata_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --video_folder=$VIDEO_FOLDER \
    --videoclipxl_score_metadata_path $VIDEOCLIPXL_SCORE_SAVED_PATH \
    --min_videoclipxl_score $MIN_VIDEOCLIPXL_SCORE \
    --saved_path=$TRAIN_SAVED_PATH