#!/bin/bash

## JOB INFO
#SBATCH --job-name=fsdp+sAC+compile+bf16+72B
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

## NODE CONFIGURATION
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --time=00:50:00


## ENV ACTIVATION
module purge
module load slurm/slurm/24.11

## CODE EXECUTION
set -x
export NCCL_MNNVL_ENABLE=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=WARN
ulimit -s 8192  # issue with thread stack when using a lot of GPUs

time srun singularity exec \
    --nv \
    --bind $WORK \
    $WORK/test_multi_noeuds/nemo_2509.sif \
    python -u $WORK/test_multi_noeuds/Democratizing-LLM-FT/FSDP_sAC.py \
        --batch-size 2 \
        --epochs 1 \
        --grad-acc 1 \
        --test \
        --display-optimizer-dtype \
        --dataset-path /lustre/work/sos/ssos027/test_multi_noeuds/Democratizing-LLM-FT/dataset/tulu-3-sft-mixture \
        --model-path $WORK/test_multi_noeuds/Democratizing-LLM-FT/model/Qwen2.5-72B-Instruct \
        --selective-activation-checkpointing 1 \
        --compile

date
