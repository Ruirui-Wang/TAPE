#!/bin/sh
#SBATCH --time=3-00:00:00
#SBATCH --partition=cpuonly
#SBATCH --job-name=gnn_wb



#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/batch

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu

source /hkfs/home/project/hk-project-test-p0022257/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate ss
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE/core/gcns

device_list=(0 1 2 3)
data=pubmed #arxiv_2023
model_list=(GAT_Variant,  GCN_Variant, SAGE_Variant,  GIN_Variant)

for index in "${!model_list[@]}"; do
    model_list=(GAT_Variant  GCN_Variant SAGE_Variant  GIN_Variant)
    for device in {0..3}; do
        model=${model_list[$device]}
        # echo "Running on data $data with device cuda:$device and model $model"
        echo "python final_gnn_tune.py --device cpu --data $data --model $model --epochs 2000 &"
        #debug
        python final_gnn_tune.py --device cpu --data $data --model $model --epochs 2000 &
        #run
        #python final_gnn_tune.py --device cuda:$device --data $data --model $model --epochs 2000 &

        while [ "$(jobs -r | wc -l)" -ge 4 ]; do
            sleep 1
        done
    done
    echo "round $index"
done


# python final_gnn_tune.py --device cuda:0 --data arxiv_2023 --model GIN_Variant --epochs 2000 --wandb True
        