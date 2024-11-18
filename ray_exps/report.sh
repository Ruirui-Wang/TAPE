#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=arxiv_afterlcc
#SBATCH --mem-per-cpu=1600mb

#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --account=hk-project-test-p0022257  # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-subgraph_training/TAPE

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wangruirui45@outlook.com

# Request GPU resources
#SBATCH --gres=gpu:1
source /hkfs/home/project/hk-project-test-p0022257/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate TAG-LP
cd /hkfs/work/workspace/scratch/cc7738-subgraph_training/TAPE
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12
export HUGGINGFACE_HUB_TOKEN=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

# python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 50 --cfg ./core/yamls/cora/lms/llama.yaml --repeat 1 --product dot
python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 50 --cfg ./core/yamls/arxiv_2023/lms/llama.yaml --repeat 1 --product dot

# python ./core/gcns/ncn_main.py --cfg ./core/yamls/arxiv_2023/gcns/ncn.yaml --data arxiv_2023 --device cuda:0 --epochs 100 --repeat 1
# python ./core/gcns/ncn_main.py --cfg ./core/yamls/cora/gcns/ncn.yaml --data cora --device cuda:0 --epochs 100 --repeat 1

# python ./core/gcns/ncn_main.py --cfg ./core/yamls/arxiv_2023/gcns/ncnc.yaml --data arxiv_2023 --device cuda:0 --epochs 100 --repeat 1
# python ./core/gcns/ncn_main.py --cfg ./core/yamls/cora/gcns/ncnc.yaml --data cora --device cuda:0 --epochs 100 --repeat 1


