#!/bin/bash

## JOB INFO
#SBATCH --job-name=linaro_map_test
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
#SBATCH --array=0


## ENV ACTIVATION
module purge
module load singularity
module load linaro-forge/25.1.2

## JOB ARRAY INDEX → AC value
AC_VALUES=(0.0 0.0 0.0)
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
export FORGE_STOP_AT_MAIN=1
export FORGE_WRAPPER_COMPILE=0
export FORGE_MPI_WRAPPER=0
export UCX_MEM_EVENTS=no

export ALLINEA_MPI_INIT=main
export ALLINEA_HOLD_MPI_INIT=1
time map --log logfile.xm --debug --profile --start-after=30 --mpi=slurm -n $SLURM_NTASKS --mpiargs="-n $SLURM_NTASKS singularity exec --nv --env MPICC=mpicc --bind $WORK/LLM-FT-IDRIS-Benchmark,$DSDIR,/lustre/fshomisc/sys/common/linaro/forge/25.1.2 $SINGULARITY_ALLOWED_DIR/nemo_2509.sif" \
    python -u $WORK/LLM-FT-IDRIS-Benchmark/FSDP_sAC.py \
        --batch-size 1 \
        --seq-length 512 \
        --epochs 1 \
        --grad-acc 8 \
        --test \
        --test-nsteps 1 \
        --display-optimizer-dtype \
        --enable-flop-counter \
        --dataset-path $WORK/LLM-FT-IDRIS-Benchmark/dataset/tulu-3-sft-mixture \
        --model-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct \
        --selective-activation-checkpointing $AC \
        --compile

date
