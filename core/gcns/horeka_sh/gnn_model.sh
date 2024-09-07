#!/bin/sh
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=152
#SBATCH --partition=accelerated
#SBATCH --job-name=gnn_wb
#SBATCH --mem=501600mb
#BATCH  --cpu-per-gpu=38
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --gres=gpu:4
#SBATCH --account=hk-project-test-p0022257  # specify the project group


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/batch

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu

source /hkfs/home/project/hk-project-test-p0022257/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate EAsF
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns


device_list=(0 1 2 3)
data_list=(pubmed)  #pubmed arxiv_2023
model_list=(GAT GAE GraphSage VGAE)

# Assuming device_list, model_list, and data_list are defined and populated

for index in "${!data_list[@]}"; do
    data=${data_list[$index]}
    models=("GAT" "GAE" "GraphSage" "VGAE")
    for device in {0..3}; do
        model=${models[$device]}
        if [ -z "$data" ]; then
            echo "Skipping round $index due to missing data"
            continue
        fi
        echo "Running on data $data with device cuda:$device and model $model"
        python universal_tune_heart.py --device cuda:$device --data $data --model $model --epochs 10000 > "${model}-${device}-${data}-universal_tune_heart_output.txt" 
        python universal_tune.py --device cuda:$device --data $data --model $model --epochs 10000 > "${model}-${device}-${data}-universal_tune_heart_output.txt" 
        
        while [ "$(jobs -r | wc -l)" -ge 4 ]; do
            sleep 1
        done
    done
    echo "round $index"
done

