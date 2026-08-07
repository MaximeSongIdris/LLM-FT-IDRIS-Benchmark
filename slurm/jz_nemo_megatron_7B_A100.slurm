#!/bin/bash

## JOB INFO
#SBATCH --job-name=nemo+megatron+7B+4dp+4tp+2bs+ovl+a100
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

## NODE CONFIGURATION
#SBATCH --constraint=a100
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --account=sos@a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=01:00:00
#SBATCH --array=0


## ENV ACTIVATION
module purge
module load singularity

## CODE EXECUTION
set -x
export NCCL_MNNVL_ENABLE=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,COLL,NET,GRAPH,TUNING,NVLS,PROXY
export NCCL_DEBUG_TIMESTAMP_LEVELS=WARN,INFO
export NCCL_DEBUG_FILE=slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_%p.nccl
ulimit -s 8192  # issue with thread stack when using a lot of GPUs

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export PYTHONUNBUFFERED=1
export HF_HOME=$WORK/LLM-FT-IDRIS-Benchmark/.cache
export CUDA_LAUNCH_BLOCKING=0
time srun singularity exec \
    --nv \
    --bind $WORK/LLM-FT-IDRIS-Benchmark,$DSDIR \
    $SINGULARITY_ALLOWED_DIR/nemo_2509.sif \
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        $WORK/LLM-FT-IDRIS-Benchmark/nemo_megatron.py \
            --global-batch-size 64 \
            --batch-size 2 \
            --seq-length 4096 \
            --epochs 1 \
            --test \
            --test-nsteps 100 \
            --no-enable-flop-counter \
            --no-pytorch-profiler \
            --dataset-root ./sft_megatron_data \
            --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct \
            --nemo-ckpt-path ./nemo_ckpt \
            --no-activation-checkpointing \
            --no-fp8 \
            --devices-per-node $SLURM_GPUS_ON_NODE \
            --num-nodes $SLURM_NNODES \
            --dp-size 1 \
            --pp-size 2 \
            --tp-size 8 \
            --cp-size 1

date
