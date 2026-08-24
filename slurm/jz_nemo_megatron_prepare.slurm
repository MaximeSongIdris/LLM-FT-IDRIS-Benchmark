#!/bin/bash

## JOB INFO
#SBATCH --job-name=nemo+megatron+prepare
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

## NODE CONFIGURATION
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --account=sos@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=01:00:00


## ENV ACTIVATION
module purge
module load singularity

## CODE EXECUTION
set -x

export PYTHONUNBUFFERED=1
export HF_HOME=$WORK/LLM-FT-IDRIS-Benchmark/.cache

# Only requires to do it once in a single process
# Convert HF ckpt into Megatron ckpt
#singularity exec \
#    --nv \
#    --bind $WORK/LLM-FT-IDRIS-Benchmark,$DSDIR \
#    $SINGULARITY_ALLOWED_DIR/nemo_2509.sif \
#    python $WORK/LLM-FT-IDRIS-Benchmark/nemo_megatron.py \
#            --prepare \
#            --dataset-path $WORK/LLM-FT-IDRIS-Benchmark/dataset/tulu-3-sft-mixture \
#            --dataset-root ./sft_megatron_data \
#            --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct \
#            --nemo-ckpt-path ./nemo_ckpt

# Convert HF ckpt into Megatron ckpt
singularity exec \
    --nv \
    --bind $WORK/LLM-FT-IDRIS-Benchmark,$DSDIR \
    $SINGULARITY_ALLOWED_DIR/nemo_2509.sif \
    python $WORK/LLM-FT-IDRIS-Benchmark/nemo_megatron.py \
            --prepare \
            --dataset-path $WORK/LLM-FT-IDRIS-Benchmark/dataset/tulu-3-sft-mixture \
            --dataset-root ./sft_megatron_data \
            --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-72B-Instruct \
            --nemo-ckpt-path ./nemo_ckpt2

date
