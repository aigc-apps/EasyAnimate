export VIDEO_FOLDER="datasets/panda_70m/before_vcut/"
export OUTPUT_FOLDER="datasets/panda_70m/train/"

# Cut raw videos
python scenedetect_vcut.py \
    $VIDEO_FOLDER \
    --threshold 10 20 30 \
    --frame_skip 0 1 2 \
    --min_seconds 3 \
    --max_seconds 10 \
    --save_dir $OUTPUT_FOLDER