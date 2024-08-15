import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

import pandas as pd
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch 
from matplotlib import pyplot as plt
from visual import load_csv
from torch_geometric import utils
from torch_geometric.utils import to_torch_coo_tensor, coalesce 
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from core.heuristic.lsf import CN, AA, RA, InverseRA
from core.heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close, SymPPR
from core.data_utils.load import load_data_lp
from core.data_utils.lcc import use_lcc
from core.graphgps.utility.utils import init_cfg_test
from visual import (find_opt_thres, 
                    get_metric_invariant, 
                    load_results, 
                    plot_pos_neg_histogram, 
                    tensor_to_csr_matrix, 
                    eval_mix_heuristic)


notebook_dir = os.getcwd()  
target_dir = os.path.abspath(os.path.join(notebook_dir, '..'))

sys.path.insert(0, target_dir)

evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name='ogbl-citation2')

# Example usage
FILE_PATH = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/educational_demo/'
ncnc_cora_path = FILE_PATH + 'err_ncnc_llama/ncnc-arxiv_2023_AUC_0.9701_MRR_0.2946.csv'

# optimal threshold detection
P1, P2, pos_index, neg_index = load_results(ncnc_cora_path)
best_thres, best_acc, pos_pred, neg_pred = find_opt_thres(P1, P2)

plot_pos_neg_histogram(P1, P2, best_thres)
plot_pos_neg_histogram(pos_pred, neg_pred, best_thres)

cfg = init_cfg_test()
cfg.data.name = 'arxiv_2023'
splits, text, data = load_data_lp[cfg.data.name](cfg.data)
# type 2 predict no when yes 

k_list  = [0.1, 0.2, 0.3, 0.5, 1]
mrr_pos2neg, mrr_neg2pos, result_auc_test, pos_edge_index_err, pos_rank_err, neg_edge_index_err, neg_rank_err = get_metric_invariant(P1, pos_index, P2, neg_index, k_list)

# pos
norm_feat_pos = eval_mix_heuristic(data, pos_edge_index_err)

feature_names = ['CN', 'AA', 'RA', 'InverseRA', 'Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR', 'feat_pro']
index_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
plt.figure(figsize=(8, 8))
plt.imshow(norm_feat_pos, cmap='Greens', aspect='auto')
plt.colorbar()
plt.xticks(ticks=np.arange(len(index_names)), labels=index_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Feature Map Heatmap')
plt.tight_layout()
plt.savefig('pos_err_map.png')


norm_feat_neg = eval_mix_heuristic(data, neg_edge_index_err)
feature_names = ['CN', 'AA', 'RA', 'InverseRA', 'Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR', 'feat_pro']
index_names = ['0']
plt.figure(figsize=(8, 8))
plt.imshow(norm_feat_neg, cmap='Greens', aspect='auto')
plt.colorbar()
plt.xticks(ticks=np.arange(len(index_names)), labels=index_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Feature Map Heatmap')
plt.tight_layout()
plt.savefig('neg_err_map.png')
plt.close()

type_2 = pos_index[pos_pred == 0]
print(f'{len(type_2)} type 2 errors are detected.')

# type 1 predict yes when no
type_1 = neg_index[neg_pred == 1]
print(f'{len(type_1)} type 1 errors are detected.')

for row in type_1:
    print(f'Source {row[0]}: {text[row[0]][:80]}, \n target {row[1]}: {text[row[1]][:80]}')
    print('-----------------------------------')
print('such citation doesnt exist in the dataset')

for row in type_2:
    print(f'Source {row[0]}: {text[row[0]][:80]}, \n target {row[1]}: {text[row[1]][:80]}')
    print('-----------------------------------')
print('such citation doesnt exist in the dataset')



norm_feat_pos = eval_mix_heuristic(data, type_1)
feature_names = ['CN', 'AA', 'RA', 'InverseRA', 'Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR', 'feat_pro']
index_names = [str(i) for i in range(type_1.shape[0])]
plt.figure(figsize=(8, 8))
plt.imshow(norm_feat_pos, cmap='Greens', aspect='auto')
plt.colorbar()
plt.xticks(ticks=np.arange(len(index_names)), labels=index_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Feature Map Heatmap')
plt.tight_layout()
plt.savefig('pos_err_type1.png')



norm_feat_pos = eval_mix_heuristic(data, type_2)
feature_names = ['CN', 'AA', 'RA', 'InverseRA', 'Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR', 'feat_pro']
index_names = [str(i) for i in range(type_2.shape[0])]
plt.figure(figsize=(8, 8))
plt.imshow(norm_feat_pos, cmap='Greens', aspect='auto')
plt.colorbar()
plt.xticks(ticks=np.arange(len(index_names)), labels=index_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Feature Map Heatmap')
plt.tight_layout()
plt.savefig('pos_err_type2.png')