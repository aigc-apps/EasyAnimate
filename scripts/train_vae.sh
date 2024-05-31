torchrun --master_addr=127.0.0.1 --master_port=6006 --nnodes=1 --node_rank=0 --nproc_per_node=1 scripts/train_vae.py \
    -t \
    -b easyanimate/vae/configs/autoencoder/autoencoder_kl_32x32x4_slice.yaml \
    --scale_lr=False \
    --resume="True"