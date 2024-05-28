CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch video_frame_quality.py \
    --video_metadata_path=/mnt_wg/huangkunzhe.hkz/dataset/shot2story/videos_shots/meta_file_info.jsonl \
    --video_folder=/mnt_wg/huangkunzhe.hkz/dataset/shot2story/videos_shots/data/ \
    --video_path_column=video_path \
    --metrics=AestheticScore \
    --saved_freq=10 \
    --saved_path=/mnt/nas/huangkunzhe.hkz/code/EasyAnimate/easyanimate/video_caption/test/aesthetic_score_shot2story.jsonl \
    --batch_size=8

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch compute_text_score.py \
    --video_metadata_path=/mnt_wg/huangkunzhe.hkz/dataset/shot2story/videos_shots/meta_file_info.jsonl \
    --video_folder=/mnt_wg/huangkunzhe.hkz/dataset/shot2story/videos_shots/data/ \
    --video_path_column="video_path" \
    --saved_freq=10 \
    --saved_path=/mnt/nas/huangkunzhe.hkz/code/EasyAnimate/easyanimate/video_caption/test/text_score_shot2story.jsonl \
    --asethetic_score_metadata_path /mnt/nas/huangkunzhe.hkz/code/EasyAnimate/easyanimate/video_caption/test/aesthetic_score_shot2story.jsonl

python compute_motion_score.py \
    --video_metadata_path=/mnt_wg/huangkunzhe.hkz/dataset/shot2story/videos_shots/meta_file_info.jsonl \
    --video_folder=/mnt_wg/huangkunzhe.hkz/dataset/shot2story/videos_shots/data/ \
    --video_path_column="video_path" \
    --saved_freq=10 \
    --saved_path=/mnt/nas/huangkunzhe.hkz/code/EasyAnimate/easyanimate/video_caption/test/motion_score_shot2story.jsonl \
    --n_jobs=8 \
    --asethetic_score_metadata_path /mnt/nas/huangkunzhe.hkz/code/EasyAnimate/easyanimate/video_caption/test/aesthetic_score_shot2story.jsonl \
    --text_score_metadata_path /mnt/nas/huangkunzhe.hkz/code/EasyAnimate/easyanimate/video_caption/test/text_score_shot2story.jsonl