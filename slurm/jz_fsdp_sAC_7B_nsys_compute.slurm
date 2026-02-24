#!/bin/bash

## JOB INFO
#SBATCH --job-name=nsight-compute
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

## NODE CONFIGURATION
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --account=sos@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00


## ENV ACTIVATION
module purge
module load singularity

## CODE EXECUTION
set -x
export NCCL_MNNVL_ENABLE=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=WARN
ulimit -s 8192  # issue with thread stack when using a lot of GPUs

export HF_HOME=$WORK/LLM-FT-IDRIS-Benchmark/.cache
time srun singularity exec \
    --nv \
    --bind $WORK/LLM-FT-IDRIS-Benchmark,$DSDIR \
    $SINGULARITY_ALLOWED_DIR/nemo_2509.sif \
    bash -c '
    if [ "$SLURM_PROCID" -eq 0 ]; then
        ncu --kernel-name regex:"cudnn_generated_fort_native_sdpa_sm90_flash" \
            --launch-count 10 \
            --set full \
            -o ncu_report/bs1 \
            python -u $WORK/LLM-FT-IDRIS-Benchmark/FSDP_sAC_nsight_sys.py \
                --batch-size 1 \
                --epochs 1 \
                --grad-acc 4 \
                --test \
                --test-nsteps 10 \
                --display-optimizer-dtype \
                --dataset-path $WORK/LLM-FT-IDRIS-Benchmark/dataset/tulu-3-sft-mixture \
                --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct \
                --selective-activation-checkpointing 0.0 \
                --compile
    else
        python -u $WORK/LLM-FT-IDRIS-Benchmark/FSDP_sAC_nsight_sys.py \
            --batch-size 2 \
            --epochs 1 \
            --grad-acc 2 \
            --test \
            --test-nsteps 10 \
            --display-optimizer-dtype \
            --dataset-path $WORK/LLM-FT-IDRIS-Benchmark/dataset/tulu-3-sft-mixture \
            --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct \
            --selective-activation-checkpointing 0.0 \
            --compile
    fi
    '

date
