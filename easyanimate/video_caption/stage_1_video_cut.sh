VIDEO_FOLDER="datasets/panda_70m/videos/data/"
META_FILE_PATH="datasets/panda_70m/videos/meta_file_info.jsonl"
META_SCENE_FOLDER="datasets/panda_70m/videos/meta_scene_info/"
META_SCENE_PATH="datasets/panda_70m/videos/meta_scene_info.jsonl"
OUTPUT_FOLDER="datasets/panda_70m/video_clips/data/"
RESOLUTION_THRESHOLD=$((512*512))
# Adjust the n_jobs parameter based on the actual number of CPU cores in the machine.


python -m utils.get_meta_file \
    --video_folder $VIDEO_FOLDER \
    --saved_path $META_FILE_PATH

python cutscene_detect.py \
    --video_metadata_path $META_FILE_PATH \
    --video_folder $VIDEO_FOLDER \
    --saved_folder $META_SCENE_FOLDER \
    --n_jobs 32

python -m utils.gather_jsonl \
    --meta_folder $META_SCENE_FOLDER \
    --meta_file_path $META_SCENE_PATH \
    --n_jobs 64

# It consumes more CPU compared to the above two operations.
python video_splitting.py \
    --video_metadata_path $META_SCENE_PATH \
    --video_folder $VIDEO_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --n_jobs 16 \
    --resolution_threshold $RESOLUTION_THRESHOLD