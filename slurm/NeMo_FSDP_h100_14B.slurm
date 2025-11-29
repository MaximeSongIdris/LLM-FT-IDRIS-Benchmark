#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/NeMo_FSDP_h100_14B_%j.out 
#SBATCH --error=logs/err/NeMo_FSDP_h100_14B_%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread 
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --account=sos@h100

## load module 
module purge
module load arch/h100
#module load nemo/2.5.2
module load singularity

## cache dir
export HF_DATASETS_CACHE="/lustre/fsn1/projects/idris/sos/ssos040/HF_CACHE/tulu3/h100_14B"

export SINGULARITY_BINDPATH="$WORK,$ALL_CCFRWORK,$DSDIR,$SCRATCH,$ALL_CCFRSCRATCH,$JOBSCRATCH,/lustre/fswork/dataset"

## Distribution setup
## Distribution setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# detect the IB trouble
export NCCL_DEBUG=WARN


## launch script on every task 
set -x
time srun singularity exec --nv --bind $SINGULARITY_BINDPATH \
    $SINGULARITY_ALLOWED_DIR/nemo2506_amd.sif \
    python -u nemo_TPFSDP2.py --model Qwen/Qwen2.5-14B-Instruct \
                             --use-te-optimizer \
                             --strategy fsdp2 \
                             --liger \
                             --devices 4 \
                             --num-nodes 16 \
                             --tp-size 1 \
                             --cp-size 1 \
                             --batch-size 2 \
                             --enable-grad-ckpt \

date

