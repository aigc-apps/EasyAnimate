META_FILE_PATH="datasets/panda_70m/videos_clips/data/meta_file_info.jsonl"
VIDEO_FOLDER="datasets/panda_70m/videos_clips/data/"
MOTION_SAVED_PATH="datasets/panda_70m/videos_clips/meta_motion_info.jsonl"
MIN_MOTION_SCORE=2
VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b.jsonl"
REWRITTEN_VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b_rewritten.jsonl"
VIDEOCLIPXL_SCORE_SAVED_PATH="datasets/panda_70m/meta_caption_info_vila_8b_rewritten_videoclipxl.jsonl"
MIN_VIDEOCLIPXL_SCORE=0.20
TRAIN_SAVED_PATH="datasets/panda_70m/train_panda_70m.json"
# Manually download OpenGVLab/InternVL2-40B-AWQ to INTERNVL2_MODEL_PATH.
# You can also download OpenGVLab/InternVL2-2B-AWQ InternVL2-8B-AWQ InternVL2-26B-AWQ or InternVL2-Llama3-76B-AWQ 
# This a trade-off between recaption quality and speed.
INTERNVL2_MODEL_PATH="OpenGVLab/InternVL2-40B-AWQ"
# Manually download meta-llama/Meta-Llama-3.1-70B-Instruct to REWRITE_MODEL_PATH.
# You can also download meta-llama/Meta-Llama-3.1-8B-Instruct Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct.
REWRITE_MODEL_PATH="meta-llama/Meta-Llama-3.1-70B-Instruct"

# Use InternVL2-AWQ to perform recaptioning.
# Adjust the num_workers and batch size parameter based on the machine's computing resources to achieve maximum GPU utilization.
python3 internvl2_video_recaptioning.py \
    --video_metadata_path ${META_FILE_PATH} \
    --video_folder ${VIDEO_FOLDER} \
    --model_path ${INTERNVL2_MODEL_PATH} \
    --saved_path $VIDEO_CAPTION_SAVED_PATH \
    --saved_freq 1 \
    --num_workers 4 \
    --batch_size 128 \
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
# Adjust the num_workers and batch size parameter based on the machine's computing resources to achieve maximum GPU utilization.
accelerate launch compute_video_quality.py \
    --video_metadata_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --caption_column "caption" \
    --video_folder $VIDEO_FOLDER \
    --frame_sample_method uniform \
    --num_sampled_frames 8 \
    --metrics VideoCLIPXLScore \
    --num_workers 4 \
    --batch_size 16 \
    --saved_path $VIDEOCLIPXL_SCORE_SAVED_PATH \
    --saved_freq 10

# Get the final train file.
python filter_meta_train.py \
    --caption_metadata_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --video_folder $VIDEO_FOLDER \
    --videoclipxl_score_metadata_path $VIDEOCLIPXL_SCORE_SAVED_PATH \
    --min_videoclipxl_score $MIN_VIDEOCLIPXL_SCORE \
    --saved_path $TRAIN_SAVED_PATH