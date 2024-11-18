#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpuonly
#SBATCH --job-name=vis_adj
#SBATCH --mem-per-cpu=1600mb

#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --account=hk-project-test-p0022257  # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-subgraph_training/TAPE

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wangruirui45@outlook.com


source /hkfs/home/project/hk-project-test-p0022257/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate TAG_LP
cd /hkfs/work/workspace/scratch/cc7738-subgraph_training/TAPE
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12




python ./core/metrics_analysis/remap_and_visualize.py