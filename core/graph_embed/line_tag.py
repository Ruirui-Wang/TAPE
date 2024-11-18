import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import scipy.sparse as ssp
import torch
import argparse
import wandb
import time
import matplotlib.pyplot as plt
import networkx as nx
import csv
from sklearn.linear_model import LogisticRegression
from ogb.linkproppred import Evaluator
from torch.utils.tensorboard import SummaryWriter
# Custom imports
from line_tf import LINE
from tune_utils import save_parameters
from heuristic.eval import get_metric_score
from data_utils.load import load_graph_lp as data_loader
from data_utils.graph_stats import plot_coo_matrix, construct_sparse_adj
from data_utils.load_data_lp import get_edge_split
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel
from torch_geometric.utils import to_scipy_sparse_matrix

# Set the file path for the project
FILE_PATH = get_git_repo_root_path() + '/'

from graph_embed.tune_utils import get_git_repo_root_path
import wandb
from networkx import from_scipy_sparse_matrix as from_scipy_sparse_array

def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    return parser.parse_args()

def set_device():
    """Set the PyTorch device based on availability."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        return 'cuda'
    return 'cpu'

def prepare_data(cfg):
    """Load and prepare the dataset."""
    if cfg.data.name == 'cora':
        dataset, _ = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'pubmed':
        dataset = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'arxiv_2023':
        dataset = data_loader[cfg.data.name]()
    undirected = dataset.is_undirected()
    splits = get_edge_split(dataset, undirected, cfg.data.device, cfg.data.split_index[1], cfg.data.split_index[2], cfg.data.include_negatives, cfg.data.split_labels)
    return dataset, splits

def create_graph(splits):
    """Construct the graph from the dataset splits."""
    full_edge_index = splits['test'].edge_index
    adj = to_scipy_sparse_matrix(full_edge_index)
    G = from_scipy_sparse_array(adj)
    return G

def train_line_model(G, cfg, device):
    """Train the LINE model on the graph."""
    model = LINE(G, embedding_size=96, order='all', lr=0.01)
    start = time.time()
    model.train(batch_size=2048, epochs=cfg.model.line.epoch, verbose=2)
    end = time.time()
    embeddings = model.get_embeddings()
    
    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}')
    params = [root, model, start, end, cfg.model.line.epoch]
    
    npz_file = os.path.join(root, f'{cfg.data.name}_embeddings.npz')
    if isinstance(embeddings, dict):
        embeddings_str_keys = {str(key): value for key, value in embeddings.items()}
        np.savez(npz_file, **embeddings_str_keys)
    else:
        np.savez(npz_file, embeddings=embeddings)
    return model, embeddings, params

def evaluate_model(splits, cfg, embed, params):
    """Evaluate the trained model using logistic regression."""
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    X_train = np.multiply(embed[X_train_index[:, 1]], embed[X_train_index[:, 0]])
    
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    X_test = np.multiply(embed[X_test_index[:, 1]], embed[X_test_index[:, 0]])

    start_eval = time.time()
    clf = LogisticRegression(solver='lbfgs', max_iter=cfg.model.line.max_iter, multi_class='auto')
    end_eval = time.time()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)
    acc = clf.score(X_test, y_test)
    
    root, model, start_train, end_train, epochs = params
    save_parameters(root, model, start_train, end_train, epochs, start_eval, end_eval, f"{cfg.data.name}_model_parameters.csv")
    return y_pred, acc, y_test

def save_results(y_pred, y_test, acc, cfg, run_name):
    """Save the evaluation results."""
    method = cfg.model.type
    
    plt.figure()
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='test')
    plt.savefig(f'{method}_pred.png')

    results_acc = {f'{method}_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    result_mrr['ACC'] = acc
    
    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}', run_name)
    os.makedirs(root, exist_ok=True)

    acc_file = os.path.join(root, f'{cfg.data.name}_acc.csv')
    mrr_file = os.path.join(root, f'{cfg.data.name}_mrr.csv')

    run_id = wandb.util.generate_id()
    append_acc_to_excel(run_id, {'line_acc': acc}, acc_file, cfg.data.name, cfg.model.type)
    append_mrr_to_excel(run_id, {'line_mrr': result_mrr}, mrr_file, cfg.data.name, cfg.model.type)

    print("Results saved.")
    return result_mrr

def main():
    """Main execution function."""
    args = parse_args()
    cfg = set_cfg(get_git_repo_root_path() + '/', args.cfg)
    
    torch.set_num_threads(cfg.num_threads)
    device = set_device()

    seeds = [1, 2, 3, 4, 5]
    mrr_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_name = f'seed_{seed}'
        writer = SummaryWriter(log_dir=os.path.join(FILE_PATH, 'runs', cfg.data.name, run_name))

        dataset, splits = prepare_data(cfg)
        G = create_graph(splits)
        model, embeddings, params = train_line_model(G, cfg, device)
        
        y_pred, acc, y_test = evaluate_model(splits, cfg, embeddings, params)
        res = save_results(y_pred, y_test, acc, cfg, run_name)

        mrr_results.append(res)

        writer.close()

    columns = {key: [d[key] for d in mrr_results] for key in mrr_results[0]}

    means = {key: np.mean(values) for key, values in columns.items()}
    variances = {key: np.var(values, ddof=1) for key, values in columns.items()}

    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}')
    mrr_file = os.path.join(root, f'{cfg.data.name}_gr_emb_res.csv')
    
    run_id = wandb.util.generate_id()
    with open(mrr_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 'MRR', 
                        'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100', 'AUC', 'AP', 'ACC'])
        
        keys = ["Hits@1", "Hits@3", "Hits@10", "Hits@20", "Hits@50", "Hits@100", "MRR", 
                "mrr_hit1", "mrr_hit3", "mrr_hit10", "mrr_hit20", "mrr_hit50", "mrr_hit100", 
                "AUC", "AP", "ACC"]

        row = [f"{run_id}_{cfg.data.name}"] + [f'{means.get(key, 0) * 100:.2f} ± {variances.get(key, 0) * 100:.2f}' for key in keys]

        writer.writerow(row)
    
    file_path = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}/{cfg.data.name}_model_parameters.csv')
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        data = []
        for row in reader:
            data.append([float(value) for value in row[1:]])

        rows = np.array(data)

    means = np.mean(rows, axis=0)
    mean_row = ['Mean'] + [f'{mean:.6f}' for mean in means]
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(mean_row)

if __name__ == "__main__":
    main()