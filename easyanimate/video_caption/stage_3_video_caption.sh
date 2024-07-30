export MAIN_PROCESS_PORT=9991
export VIDEO_FOLDER="datasets/panda_70m/train/"
export MOTION_SCORE_META_PATH="datasets/panda_70m/train.jsonl"
export VIDEO_FRAME_CAPTION_PATH="datasets/panda_70m/frame_caption.jsonl"
export VIDEO_CAPTION_PATH="datasets/panda_70m/summary_caption.jsonl"
export LAST_JSON_PATH="datasets/panda_70m/train_panda_70m.json"
export MODEL_PATH="/PATH/TO/Llama-3-VILA1.5-8b-AWQ"
export QUANT_PATH=${MODEL_PATH}/llm/llama-3-vila1.5-8b-w4-g128-awq-v2.pt

HF_ENDPOINT=https://hf-mirror.com accelerate launch --main_process_port ${MAIN_PROCESS_PORT} vila_caption_video.py \
    --video_metadata_path ${MOTION_SCORE_META_PATH} \
    --video_folder ${VIDEO_FOLDER} \
    --model_path ${MODEL_PATH} \
    --quant_path ${QUANT_PATH} \
    --precision "W4A16" \
    --saved_path VIDEO_CAPTION_PATH \
    --saved_freq 1

python convert_jsonl_to_json.py \
    --video_folder=$VIDEO_FOLDER \
    --jsonl_load_path=$VIDEO_CAPTION_PATH \
    --save_path=$LAST_JSON_PATH