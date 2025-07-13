#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
# make sure the checkpoint is under `MedSAM2/checkpoints/sam2.1_hiera_tiny.pt`

# # baseline here
# config=configs/sam2.1_hiera_tiny512_FLARE_baseline.yaml
# output_path=./exp_log/MedSAM2_FLARE25_RECIST_baseline

# dev test
config=configs/sam2.1_hiera_tiny_finetune512_dev.yaml
output_path=./exp_log/MedSAM2_finetune512_dev_1

# Function to run the training script
CUDA_VISIBLE_DEVICES=0 python training/train.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1 
        # --master-addr $MASTER_ADDR \
        # --main-port $MASTER_PORT

echo "training done"