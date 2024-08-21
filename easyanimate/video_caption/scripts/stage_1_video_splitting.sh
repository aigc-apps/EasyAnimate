VIDEO_FOLDER="datasets/panda_70m/videos/data/"
META_FILE_PATH="datasets/panda_70m/videos/meta_file_info.jsonl"
SCENE_FOLDER="datasets/panda_70m/videos/meta_scene_info/"
SCENE_SAVED_PATH="datasets/panda_70m/videos/meta_scene_info.jsonl"
OUTPUT_FOLDER="datasets/panda_70m/videos_clips/data/"
RESOLUTION_THRESHOLD=$((512*512))

# Set the duration range of video clips.
export MIN_SECONDS=3
export MAX_SECONDS=10

# Save all video names in a video folder as a meta file.
python -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH

# Perform scene detection on the video dataset.
# Adjust the n_jobs parameter based on the actual number of CPU cores in the machine.
python cutscene_detect.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --saved_folder $SCENE_FOLDER \
    --n_jobs 32

# Gather all scene jsonl files to a single scene jsonl file.
# Adjust the n_jobs parameter based on the actual I/O speed in the machine.
python -m utils.gather_jsonl \
    --meta_folder $SCENE_FOLDER \
    --meta_file_path $SCENE_SAVED_PATH \
    --n_jobs 64

# Perform video splitting filtered by the RESOLUTION_THRESHOLD.
# It consumes more CPU computing resources compared to the above operations.
python video_splitting.py \
    --video_metadata_path $SCENE_SAVED_PATH \
    --video_folder $VIDEO_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --n_jobs 16 \
    --resolution_threshold $RESOLUTION_THRESHOLD