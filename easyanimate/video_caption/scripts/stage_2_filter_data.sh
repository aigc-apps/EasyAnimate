META_FILE_PATH="datasets/panda_70m/videos_clips/data/meta_file_info.jsonl"
VIDEO_FOLDER="datasets/panda_70m/videos_clips/data/"
VIDEO_QUALITY_SAVED_PATH="datasets/panda_70m/videos_clips/meta_quality_info_siglip.jsonl"
MIN_ASETHETIC_SCORE_SIGLIP=4.0
TEXT_SAVED_PATH="datasets/panda_70m/videos_clips/meta_text_info.jsonl"
MIN_TEXT_SCORE=0.02
MOTION_SAVED_PATH="datasets/panda_70m/videos_clips/meta_motion_info.jsonl"

python -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH

# Get the asethetic score (SigLIP) of all videos
accelerate launch compute_video_quality.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --metrics "AestheticScoreSigLIP" \
    --frame_sample_method uniform \
    --num_sampled_frames 4 \
    --saved_freq 10 \
    --saved_path $VIDEO_QUALITY_SAVED_PATH \
    --batch_size 4

# Get the text score of all videos filtered by the video quality score.
accelerate launch compute_text_score.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER  \
    --saved_freq 10 \
    --saved_path $TEXT_SAVED_PATH \
    --asethetic_score_siglip_metadata_path $VIDEO_QUALITY_SAVED_PATH \
    --min_asethetic_score_siglip $MIN_ASETHETIC_SCORE_SIGLIP

# Get the motion score of all videos filtered by the video quality score and text score.
python compute_motion_score.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --saved_freq 10 \
    --saved_path $MOTION_SAVED_PATH \
    --n_jobs 8 \
    --text_score_metadata_path $TEXT_SAVED_PATH \
    --min_text_score $MIN_TEXT_SCORE
