META_FILE_PATH="datasets/panda_70m/videos_clips/data/meta_file_info.jsonl"
VIDEO_FOLDER="datasets/panda_70m/videos_clips/data/"
CLIP_OR_DINO_MODEL_PATH="openai/clip-vit-large-patch14-336"
SEMANTIC_CONSISTENCY_SAVED_PATH="datasets/panda_70m/videos_clips/meta_consistency_info.jsonl"
MIN_SEMANTIC_CONSISTENCY_SCORE=0.8
QUALITY_SAVED_PATH="datasets/panda_70m/videos_clips/meta_quality_info_siglip.jsonl"
MIN_AESTHETIC_SCORE_SIGLIP=4.0
TEXT_SAVED_PATH="datasets/panda_70m/videos_clips/meta_text_info.jsonl"
MIN_TEXT_SCORE=0.02
MOTION_SAVED_PATH="datasets/panda_70m/videos_clips/meta_motion_info.jsonl"

python -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH

# Get the semantic consistency score of all video clips.
# Adjust the num_workers and batch size parameter based on the machine's computing resources to achieve maximum GPU utilization.
accelerate launch compute_semantic_consistency.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --model_path $CLIP_OR_DINO_MODEL_PATH \
    --frame_sample_method keyframe+first \
    --batch_size 16 \
    --num_workers 4 \
    --saved_freq 10 \
    --saved_path $SEMANTIC_CONSISTENCY_SAVED_PATH

# Get the aesthetic score (SigLIP) of all videos filtered by the semantic consistency score.
# Adjust the num_workers and batch size parameter based on the machine's computing resources to achieve maximum GPU utilization.
accelerate launch compute_video_quality.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --metrics AestheticScoreSigLIP \
    --frame_sample_method uniform \
    --num_sampled_frames 4 \
    --batch_size 16 \
    --num_workers 4 \
    --saved_freq 10 \
    --saved_path $QUALITY_SAVED_PATH \
    --semantic_consistency_score_metadata_path $SEMANTIC_CONSISTENCY_SAVED_PATH \
    --min_semantic_consistency_score $MIN_SEMANTIC_CONSISTENCY_SCORE

# Get the text score of all videos filtered by the semantic consistency score and video quality score.
accelerate launch compute_text_score.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER  \
    --saved_freq 10 \
    --saved_path $TEXT_SAVED_PATH \
    --semantic_consistency_score_metadata_path $SEMANTIC_CONSISTENCY_SAVED_PATH \
    --min_semantic_consistency_score $MIN_SEMANTIC_CONSISTENCY_SCORE \
    --aesthetic_score_siglip_metadata_path $QUALITY_SAVED_PATH \
    --min_aesthetic_score_siglip $MIN_AESTHETIC_SCORE_SIGLIP

# Get the motion score of all videos filtered by the semantic consistency score, video quality score and text score.
# Adjust the n_jobs parameter based on the actual number of CPU cores in the machine.
python compute_motion_score.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --saved_freq 10 \
    --saved_path $MOTION_SAVED_PATH \
    --n_jobs 8 \
    --semantic_consistency_score_metadata_path $SEMANTIC_CONSISTENCY_SAVED_PATH \
    --min_semantic_consistency_score $MIN_SEMANTIC_CONSISTENCY_SCORE \
    --aesthetic_score_siglip_metadata_path $QUALITY_SAVED_PATH \
    --min_aesthetic_score_siglip $MIN_AESTHETIC_SCORE_SIGLIP \
    --text_score_metadata_path $TEXT_SAVED_PATH \
    --min_text_score $MIN_TEXT_SCORE