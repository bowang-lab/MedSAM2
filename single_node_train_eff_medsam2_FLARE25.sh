#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
# wget -c -P checkpoints https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt
# Set the configuration file path
config=configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml
output_path=./exp_log/EfficientMedSAM2_small_FLARE25_RECIST

# Run the training script
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus 2 \
        --num-nodes 1


