META_FILE_PATH="datasets/panda_70m/videos_clips/data/meta_file_info.jsonl"
VIDEO_FOLDER="datasets/panda_70m/videos_clips/data/"
SEMANTIC_CONSISTENCY_SAVED_PATH="datasets/panda_70m/videos_clips/meta_consistency_info.jsonl"
MIN_SEMANTIC_CONSISTENCY_SCORE=0.8
QUALITY_SAVED_PATH="datasets/panda_70m/videos_clips/meta_quality_info_siglip.jsonl"
MIN_AESTHETIC_SCORE_SIGLIP=4.0
TEXT_SAVED_PATH="datasets/panda_70m/videos_clips/meta_text_info.jsonl"
MIN_TEXT_SCORE=0.02
MOTION_SAVED_PATH="datasets/panda_70m/videos_clips/meta_motion_info.jsonl"
MIN_MOTION_SCORE=2
MAX_MOTION_SCORE=20
VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info.jsonl"
REWRITTEN_VIDEO_CAPTION_SAVED_PATH="datasets/panda_70m/meta_caption_info_rewritten.jsonl"
VIDEOCLIPXL_SCORE_SAVED_PATH="datasets/panda_70m/meta_caption_info_rewritten_videoclipxl.jsonl"
MIN_VIDEOCLIPXL_SCORE=0.20
TRAIN_SAVED_PATH="datasets/panda_70m/train_panda_70m.json"
# Manually download OpenGVLab/InternVL2-40B-AWQ to CAPTION_MODEL_PATH.
# You can also download OpenGVLab/InternVL2-2B-AWQ InternVL2-8B-AWQ InternVL2-26B-AWQ or InternVL2-Llama3-76B-AWQ 
# This a trade-off between recaption quality and speed.
CAPTION_MODEL_PATH="OpenGVLab/InternVL2-40B-AWQ"
# Manually download meta-llama/Meta-Llama-3.1-70B-Instruct to REWRITE_MODEL_PATH.
# You can also download meta-llama/Meta-Llama-3.1-8B-Instruct Meta-Llama-3-8B-Instruct Meta-Llama-3-70B-Instruct.
REWRITE_MODEL_PATH="meta-llama/Meta-Llama-3.1-70B-Instruct"

# Use InternVL2-AWQ to perform recaptioning.
# Adjust the num_workers and batch size parameter based on the machine's computing resources to achieve maximum GPU utilization.
python3 internvl2_video_recaptioning.py \
    --video_metadata_path ${META_FILE_PATH} \
    --video_folder ${VIDEO_FOLDER} \
    --model_path ${CAPTION_MODEL_PATH} \
    --saved_path $VIDEO_CAPTION_SAVED_PATH \
    --saved_freq 1 \
    --num_workers 4 \
    --batch_size 128 \
    --semantic_consistency_score_metadata_path $SEMANTIC_CONSISTENCY_SAVED_PATH \
    --min_semantic_consistency_score $MIN_SEMANTIC_CONSISTENCY_SCORE \
    --aesthetic_score_siglip_metadata_path $QUALITY_SAVED_PATH \
    --min_aesthetic_score_siglip $MIN_AESTHETIC_SCORE_SIGLIP \
    --text_score_metadata_path $TEXT_SAVED_PATH \
    --min_text_score $MIN_TEXT_SCORE \
    --motion_score_metadata_path $MOTION_SAVED_PATH \
    --min_motion_score $MIN_MOTION_SCORE \
    --max_motion_score $MAX_MOTION_SCORE

# Rewrite video captions (optional).
python caption_rewrite.py \
    --video_metadata_path $VIDEO_CAPTION_SAVED_PATH \
    --video_path_column "video_path" \
    --batch_size 4096 \
    --model_name $REWRITE_MODEL_PATH \
    --prompt prompt/rewrite.txt \
    --prefix '"rewritten description": ' \
    --answer_template 'your rewritten description here' \
    --max_retry_count 10 \
    --saved_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --saved_freq 1

# Compute caption-video alignment (optional).
# Adjust the num_workers and batch size parameter based on the machine's computing resources to achieve maximum GPU utilization.
accelerate launch compute_video_quality.py \
    --video_metadata_path $REWRITTEN_VIDEO_CAPTION_SAVED_PATH \
    --caption_column caption \
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