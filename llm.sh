#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=pubmed_dec
#SBATCH --mem=501600mb

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


python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/minilm.yaml --product 'euclidean'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/bert.yaml --product 'euclidean'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/e5-large.yaml --product 'euclidean'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/llama.yaml --product 'euclidean'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/minilm.yaml --product 'concat'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/bert.yaml --product 'concat'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/e5-large.yaml --product 'concat'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/llama.yaml --product 'concat'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/minilm.yaml --product 'cross'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/bert.yaml --product 'cross'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/e5-large.yaml --product 'cross'

python ./core/embedding_mlp/embedding_LLM_main.py --device cuda:0 --epochs 150 --cfg ./core/yamls/pubmed/lms/llama.yaml --product 'cross'