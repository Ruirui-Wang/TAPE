#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=citationv8-0.01-bert


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

# Define the arrays for devices and datas
devices=("cuda:0" "cuda:1" "cuda:2")  # Replace with your actual device names
datas=("citationv8" "ogbn_arxiv" "pwc_medium")  # Replace with your actual data names

# Get the length of the arrays (assuming both arrays have the same length)
length=${#devices[@]}

# Loop through the arrays
for ((i=0; i<$length; i++)); do
    device=${devices[$i]}
    data=${datas[$i]}
    echo "python ./core/embedding_finetuning/embedding_LLM_main.py --data $data --scale 0.001 --device $device --epochs 1000"
    python ./core/embedding_finetuning/embedding_LLM_main.py --data "$data" --scale 0.001 --device "$device" --epochs 1000 &
done

#python ./core/embedding/embedding_LLM_main.py --device cuda:0 --epochs 1000 --cfg ./core/yamls/arxiv_2023/lms/bert.yaml
# echo "python ./core/embedding/embedding_LLM_main.py --data citationv8 --scale 0.001 --device cuda:0 --epochs 1000"
# for device in cuda:0 cuda:1 cuda:2 cuda:3; do
#     python ./core/embedding/embedding_LLM_main.py --data citationv8 --scale 0.001 --device $device --epochs 1000
# done
