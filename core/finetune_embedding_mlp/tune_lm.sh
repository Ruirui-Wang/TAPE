#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=tune_lm


#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --account=hk-project-test-p0022257  # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_test/TAPE

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen.shao2@kit.edu

# Request GPU resources
#SBATCH --gres=gpu:1
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate TAG-LP
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_test/TAPE

# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

# multi-gpu distributed training TODO

# report 

CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/lm_trainer_tune.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/ncn.yaml
CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/lm_trainer_tune.py --cfg core/yamls/cora/lms/ft-e5-large.yaml --decoder core/yamls/cora/gcns/ncn.yaml
CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/lm_trainer_tune.py --cfg core/yamls/cora/lms/ft-mpnet.yaml --decoder core/yamls/cora/gcns/ncn.yaml


CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer_tune.py --cfg core/yamls/cora/lms/ft-mpnet.yaml --decoder core/yamls/cora/gcns/ncnc.yaml
CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer_tune.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/ncnc.yaml
CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer_tune.py --cfg core/yamls/cora/lms/ft-e5-large.yaml --decoder core/yamls/cora/gcns/ncnc.yaml

# round 2
CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/tune2.py --cfg core/yamls/cora/lms/ft-mpnet.yaml --decoder core/yamls/cora/gcns/ncn.yaml

srun --nodes=1 --ntasks=1 --gres=gpu:1 
CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/tune2.py --cfg core/yamls/cora/lms/ft-mpnet.yaml --decoder core/yamls/cora/gcns/ncn.yaml

srun --nodes=1 --ntasks=1 --gres=gpu:1 python core/finetune_embedding_mlp/tune2.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/ncn.yaml

srun --nodes=1 --ntasks=1 --gres=gpu:1 
python core/finetune_embedding_mlp/tune2.py --cfg core/yamls/cora/lms/ft-e5-large.yaml --decoder core/yamls/cora/gcns/ncn.yaml


# round 3
CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/tune3.py --cfg core/yamls/cora/lms/ft-mpnet.yaml --decoder core/yamls/cora/gcns/ncn.yaml
CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/tune3.py --cfg core/yamls/cora/lms/ft-minilm.yaml  --decoder core/yamls/cora/gcns/ncn.yaml
CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/tune3.py --cfg core/yamls/cora/lms/ft-e5-large.yaml --decoder core/yamls/cora/gcns/ncn.yaml


srun --nodes=1 --ntasks=1 --gres=gpu:1  CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/tune3.py --cfg core/yamls/cora/lms/ft-minilm.yaml  --decoder core/yamls/cora/gcns/ncn.yaml
srun --nodes=1 --ntasks=1 --gres=gpu:1  CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=True python core/finetune_embedding_mlp/tune3.py --cfg core/yamls/cora/lms/ft-e5-large.yaml --decoder core/yamls/cora/gcns/ncn.yaml
