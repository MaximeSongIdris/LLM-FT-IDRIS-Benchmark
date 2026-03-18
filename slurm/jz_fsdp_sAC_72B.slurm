#!/bin/bash

## JOB INFO
#SBATCH --job-name=64fsdp+1.0AC+72B+512bs
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

## NODE CONFIGURATION
#SBATCH --constraint=h100
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --account=sos@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=01:00:00
#SBATCH --array=0-2


## ENV ACTIVATION
module purge
module load singularity

## JOB ARRAY INDEX → AC value
AC_VALUES=(1.0 1.0 1.0)
AC=${AC_VALUES[$SLURM_ARRAY_TASK_ID]}

## CODE EXECUTION
set -x
export NCCL_MNNVL_ENABLE=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET,GRAPH,TUNING,NVLS,PROXY
export NCCL_DEBUG_TIMESTAMP_LEVELS=INFO,WARN
export NCCL_DEBUG_FILE=slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_%p.nccl
ulimit -s 8192  # issue with thread stack when using a lot of GPUs

export HF_HOME=$WORK/LLM-FT-IDRIS-Benchmark/.cache
time srun singularity exec \
    --nv \
    --bind $WORK/LLM-FT-IDRIS-Benchmark,$DSDIR \
    $SINGULARITY_ALLOWED_DIR/nemo_2509.sif \
    python -u $WORK/LLM-FT-IDRIS-Benchmark/FSDP_sAC.py \
        --batch-size 4 \
        --epochs 1 \
        --grad-acc 2 \
        --test \
        --test-nsteps 10 \
        --display-optimizer-dtype \
        --dataset-path $WORK/LLM-FT-IDRIS-Benchmark/dataset/tulu-3-sft-mixture \
        --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-72B-Instruct \
        --selective-activation-checkpointing $AC \
        --compile

date
