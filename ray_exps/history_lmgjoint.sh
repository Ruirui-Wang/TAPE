#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=lmgjoint

#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_german/TAPE/TAPE

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen.shao2@kit.edu

# Request GPU resources
#SBATCH --gres=gpu:1
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate TAPE
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_german/TAPE/TAPE
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/history/lms/ft-minilm.yaml --decoder core/yamls/history/gcns/ncn.yaml --repeat 2