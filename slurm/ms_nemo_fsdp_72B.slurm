#!/bin/bash

## JOB INFO
#SBATCH --job-name=nemo+72B
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

## NODE CONFIGURATION
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --time=00:50:00


## ENV ACTIVATION
module purge
module load slurm/slurm/24.11
export HF_DATASETS_CACHE="/lustre/work/sos/ssos027/cache_hf"


## CODE EXECUTION
set -x
export NCCL_MNNVL_ENABLE=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=WARN
ulimit -s 8192  # issue with thread stack when using a lot of GPUs
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export PYTHONUNBUFFERED=1
time srun singularity exec \
    --nv \
    --bind $WORK \
    $WORK/test_multi_noeuds/nemo_2509.sif \
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        $WORK/test_multi_noeuds/Democratizing-LLM-FT/nemo_TPFSDP2.py \
            --global-batch-size 32 \
            --batch-size 2 \
            --epochs 1 \
            --dataset-path /lustre/work/sos/ssos027/test_multi_noeuds/Democratizing-LLM-FT/dataset/tulu-3-sft-mixture \
            --model-path $WORK/test_multi_noeuds/Democratizing-LLM-FT/model/Qwen2.5-72B-Instruct \
            --attn-implementation flash_attention_2 \
            --compile \
            --strategy FSDP2+TP+CP \
            --devices-per-node $SLURM_GPUS_ON_NODE \
            --num-nodes $SLURM_NNODES \
            --dp-size $(($SLURM_NNODES / 2)) \
            --tp-size $SLURM_GPUS_ON_NODE \
            --cp-size 2 \

date
