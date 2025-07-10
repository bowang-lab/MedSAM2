#!/bin/bash
#SBATCH -t 7-00:0:0
#SBATCH -J medsam2-tr-tiny
#SBATCH --mem=450G
#SBATCH -c 60
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -o out_mnodes_tiny.out

export PATH=/usr/local/cuda/bin:$PATH
timestamp=$(date +"%Y%m%d-%H%M")

# Set the master node address (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=29500
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

# Print some information
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

config=configs/sam2.1_hiera_tiny_finetune512.yaml
output_path=./exp_log/mnode_tiny

# Function to run the training script
python training/train.py \
        -c $config \
        --output-path $output_path \
        --use-cluster 0 \
        --num-gpus 1 \
        --num-nodes 1 \

echo "training done"


