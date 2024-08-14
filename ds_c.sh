#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=ds_check


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

# multi-gpu distributed training TODO

#!/bin/bash



python ./core/embedding_finetuning/embedding_LLM_main.py --device cuda:0 --epochs 100 --cfg ./core/yamls/ogbn-arxiv/lms/minilm.yaml --repeat 1
python ./core/embedding_finetuning/embedding_LLM_main.py --device cuda:0 --epochs 100 --downsampling 0.01 --cfg ./core/yamls/ogbn-arxiv/lms/minilm.yaml --repeat 1
python ./core/embedding_finetuning/embedding_LLM_main.py --device cuda:0 --epochs 100 --downsampling 0.02 --cfg ./core/yamls/ogbn-arxiv/lms/minilm.yaml --repeat 1
python ./core/embedding_finetuning/embedding_LLM_main.py --device cuda:0 --epochs 100 --downsampling 0.05 --cfg ./core/yamls/ogbn-arxiv/lms/minilm.yaml --repeat 1
python ./core/embedding_finetuning/embedding_LLM_main.py --device cuda:0 --epochs 100 --downsampling 0.2 --cfg ./core/yamls/ogbn-arxiv/lms/minilm.yaml --repeat 1
python ./core/embedding_finetuning/embedding_LLM_main.py --device cuda:0 --epochs 100 --downsampling 0.5 --cfg ./core/yamls/ogbn-arxiv/lms/minilm.yaml --repeat 1


