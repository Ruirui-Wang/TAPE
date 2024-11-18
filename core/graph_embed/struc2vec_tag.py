import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import wandb
import argparse
import numpy as np
import scipy.sparse as ssp
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.graphgym.config import cfg
from ogb.linkproppred import Evaluator
from tune_utils import save_parameters
from heuristic.eval import get_metric_score
from data_utils.load_data_lp import get_edge_split
from data_utils.load import load_graph_lp as data_loader
from graphgps.utility.utils import (
    set_cfg,
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)
from ge.models.struc2vec import Struc2Vec
from networkx import from_scipy_sparse_matrix as from_scipy_sparse_array
# not maintained
# Function to parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', type=str, default='core/yamls/cora/gcns/ncn.yaml', 
                        help='Path to the configuration file.')
    parser.add_argument('--sweep', type=str, default='core/yamls/cora/gcns/ncn.yaml', 
                        help='Path to the sweep file.')
    return parser.parse_args()

# Function to set up the device
def setup_device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        return 'cuda'
    return 'cpu'

# Function to preprocess data
def preprocess_data(cfg):
    dataset, _ = data_loader[cfg.data.name](cfg)
    undirected = dataset.is_undirected()
    splits = get_edge_split(
        dataset, undirected, cfg.data.device, cfg.data.split_index[1],
        cfg.data.split_index[2], cfg.data.include_negatives, cfg.data.split_labels
    )
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset.num_nodes
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes))
    adj = to_scipy_sparse_matrix(full_edge_index)
    G = from_scipy_sparse_array(adj)
    
    return dataset, splits, G, full_A

# Function to train the Struc2Vec model
def train_model(G, cfg):
    model = Struc2Vec(
        G, walk_length=10, num_walks=80, workers=20, verbose=40,
        data=cfg.data.name, reuse=False, temp_path='./temp_path'
    )
    start = time.time()
    epochs = 5
    model.train(embed_size=128, window_size=5, workers=20, epochs=epochs)
    end = time.time()
    embed = model.get_embeddings()
    save_parameters(model, start, end, epochs)
    
    return model, embed

# Function to evaluate the model
def evaluate_model(embed, splits, cfg):
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    X_train = np.multiply(embed[X_train_index[:, 1]], embed[X_train_index[:, 0]])
    
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    X_test = np.multiply(embed[X_test_index[:, 1]], embed[X_test_index[:, 0]])
    
    clf = LogisticRegression(solver='lbfgs', max_iter=cfg.model.struc2vec.max_iter, multi_class='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)
    acc = clf.score(X_test, y_test)
    
    plt.figure()
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='test')
    plt.savefig('node2vec_pred.png')
    
    return acc, y_pred, y_test

# Function to calculate MRR
def calculate_mrr(y_pred, y_test):
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    return result_mrr

# Function to save results
def save_results(cfg, acc, result_mrr):
    root = os.path.join(get_git_repo_root_path(), 'results')
    acc_file = os.path.join(root, f'{cfg.data.name}_acc.csv')
    mrr_file = os.path.join(root, f'{cfg.data.name}_mrr.csv')
    
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    id = wandb.util.generate_id()
    results_acc = {'struc2vec_acc': acc}
    results_mrr = {'struc2vec_mrr': result_mrr}
    
    append_acc_to_excel(id, results_acc, acc_file, cfg.data.name, 'struc2vec')
    append_mrr_to_excel(id, results_mrr, mrr_file, cfg.data.name, 'struc2vec')

if __name__ == "__main__":
    args = parse_args()
    cfg = set_cfg(get_git_repo_root_path() + '/', args.cfg)
    device = setup_device()
    
    torch.set_num_threads(cfg.num_threads)
    
    dataset, splits, G, full_A = preprocess_data(cfg)
    model, embed = train_model(G, cfg)
    
    print(f"Embedding size: {embed.shape}")
    
    acc, y_pred, y_test = evaluate_model(embed, splits, cfg)
    result_mrr = calculate_mrr(y_pred, y_test)
    result_mrr['ACC'] = acc
    
    save_results(cfg, acc, result_mrr)
