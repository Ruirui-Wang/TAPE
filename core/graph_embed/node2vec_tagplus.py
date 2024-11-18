# Standard library imports
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Third-party library imports
import numba
import csv
import wandb
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from IPython import embed
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter  # Импортируем TensorBoard

# External module imports
import torch
import matplotlib.pyplot as plt
from ogb.linkproppred import Evaluator
from yacs.config import CfgNode as CN
# from torch_geometric.graphgym.cmd_args import parse_args

from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_scipy_sparse_matrix
import itertools 
import scipy.sparse as ssp
from data_utils.load import load_graph_lp as data_loader
from heuristic.eval import get_metric_score
from data_utils.load_data_lp import get_edge_split
from data_utils.lcc import construct_sparse_adj
from data_utils.graph_stats import plot_coo_matrix
from graphgps.utility.utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel,
    set_cfg,
    parse_args
)

FILE_PATH = get_git_repo_root_path() + '/'

# # Change by data_loader_graph
# data_loader = {
#     'cora': get_cora_casestudy,
#     'pubmed': get_pubmed_casestudy,
#     'arxiv_2023': get_raw_text_arxiv_2023
# }

def set_cfg(file_path, cfg_file):
    with open(file_path + cfg_file, "r") as f:
        return CN.load_cfg(f)

def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
    
def node2vec(workers, 
            adj, 
            embedding_dim, 
            walk_length, 
            walks_per_node,
            num_neg_samples, 
            p, 
            q,
            window, 
            epoch, 
            sg, 
            hs,
            min_count,
            shrink_window 
            ):
    """
    参数说明
    -------------
    adj : 图的邻接矩阵
    embedding_dim : 图嵌入的维度
    walk_length : 随机游走的长度
    walks_per_node : 每个节点采样多少个随机游走
    workers: word2vec模型使用的线程数量
    window_size: word2vec模型中使用的窗口大小
    num_neg_samples : 负样本的数量
    p: node2vec的p参数
    q: node2vec的q参数
    """
    if workers == 1:
        adj = sp.csr_matrix(adj)
        walks = _n2v_random_walk_iterator(adj.indptr,
                                    adj.indices,
                                    walk_length,
                                    walks_per_node,
                                    p,
                                    q)
        
    else:
    
        results = Parallel(n_jobs=workers, verbose=0, )(
            delayed(sample_n2v_random_walks)(adj, walk_length, num, p=p, q=q) for num in
            partition_num(walks_per_node, workers))
        walks = np.asarray(list(itertools.chain(*results)))
            
        walks = [list(map(str, walk)) for walk in walks]
    
    
    model = Word2Vec(walks, 
                     vector_size=embedding_dim, 
                     negative=num_neg_samples, 
                     compute_loss=True,
                     workers = workers,
                    )  

    embedding = model.wv.vectors[np.fromiter(map(int, model.wv.index_to_key), np.int32).argsort()] 
    
    return embedding

    
def sample_n2v_random_walks(adj, walk_length, walks_per_node, p, q):
    """
    返回值的类型
    -------
    walks : np.ndarray, shape [num_walks * num_nodes, walk_length]
        采样后的随机游走
    """
    adj = sp.csr_matrix(adj)
    random_walks = _n2v_random_walk(adj.indptr,
                                    adj.indices,
                                    walk_length,
                                    walks_per_node,
                                    p,
                                    q)
    return random_walks 


@numba.jit(nopython=True)
def _n2v_random_walk_iterator(indptr,
                    indices,
                    walk_length,
                    walks_per_node,
                    p,
                    q):
    N = len(indptr) - 1 # 节点数量

    for _ in range(walks_per_node):
        for n_iter in range(N):
            walk = np.zeros(walk_length+1, dtype=np.int32)
            walk[0] = n_iter   

            visited_set = None
            for il in range(walk_length):
                # v_iter 's neighbors
                neighbors = indices[indptr[n_iter]:indptr[n_iter+1]]
                
                sample_idx_arr = np.zeros(len(neighbors)+1, dtype=np.int32)
                sample_prob_arr = np.zeros(len(neighbors)+1, dtype=np.float32)
                
                for i, samples in enumerate(neighbors):
                    sample_idx_arr[i] = samples
                
                    if visited_set is not None:
                        if samples in visited_set:
                            sample_prob_arr[i] = 1
                        else:
                            sample_prob_arr[i] = 1/q
                    else:
                        sample_prob_arr[i] = 1/q 

                visited_set = neighbors.copy()
                sample_idx_arr[-1] = n_iter
                sample_prob_arr[-1] = 1/p
                    
                sample_prob_arr = sample_prob_arr / np.sum(sample_prob_arr)
                n_iter = random_choice(sample_idx_arr, sample_prob_arr)
                
                walk[il+1] = n_iter
            yield walk 


def _n2v_random_walk(indptr,
                    indices,
                    walk_length,
                    walks_per_node,
                    p,
                    q):
    N = len(indptr) - 1 # 节点数量
    final_walk = []
    for _ in range(walks_per_node):
        for n_iter in range(N):
            walk = np.zeros(walk_length+1, dtype=np.int32)
            walk[0] = n_iter   

            visited_set = None
            for il in range(walk_length):
                # v_iter 's neighbors
                neighbors = indices[indptr[n_iter]:indptr[n_iter+1]]
                
                sample_idx_arr = np.zeros(len(neighbors)+1, dtype=np.int32)
                sample_prob_arr = np.zeros(len(neighbors)+1, dtype=np.float32)
                
                for i, samples in enumerate(neighbors):
                    sample_idx_arr[i] = samples
                
                    if visited_set is not None:
                        if samples in visited_set:
                            sample_prob_arr[i] = 1
                        else:
                            sample_prob_arr[i] = 1/q
                    else:
                        sample_prob_arr[i] = 1/q 

                visited_set = neighbors.copy()
                sample_idx_arr[-1] = n_iter
                sample_prob_arr[-1] = 1/p
                    
                sample_prob_arr = sample_prob_arr / np.sum(sample_prob_arr)
                n_iter = random_choice(sample_idx_arr, sample_prob_arr)
                
                walk[il+1] = n_iter
            
            final_walk.append(walk)
        return np.array(final_walk)
            
@numba.jit(nopython=True)
def random_choice(arr: np.int64, p):
    """
    params
    ----------
    arr : 1d tensor
    p : probability of each element in arr
    
    返回值
    -------
    samples : sampled samples 
    adopted from blog https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
    """
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]

def eval_mrr_acc(cfg, writer=None) -> None:
    if cfg.data.name == 'cora':
        dataset, _ = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'pubmed':
        dataset = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'arxiv_2023':
        dataset = data_loader[cfg.data.name]()
    
    undirected = dataset.is_undirected()
    splits = get_edge_split(dataset, undirected, cfg.data.device,
                            cfg.data.split_index[1], cfg.data.split_index[2],
                            cfg.data.include_negatives, cfg.data.split_labels)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
    # Access individual parameters
    walk_length = cfg.model.node2vec.walk_length
    walks_per_node = cfg.model.node2vec.num_walks
    p = cfg.model.node2vec.p
    q = cfg.model.node2vec.q
    workers = cfg.model.node2vec.workers
    embed_size = cfg.model.node2vec.embed_size
    iter = cfg.model.node2vec.max_iter
    num_neg_samples = cfg.model.node2vec.num_neg_samples 
    window = cfg.model.node2vec.window
    min_count = cfg.model.node2vec.min_count
    shrink_window = cfg.model.node2vec.shrink_window
    epoch = cfg.model.node2vec.epoch
    sg = cfg.model.node2vec.sg 
    hs = cfg.model.node2vec.hs
    
    # G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
    adj = to_scipy_sparse_matrix(full_edge_index)

    embed = node2vec(workers, 
                    adj, 
                    embedding_dim=embed_size, 
                    walk_length=walk_length,
                    walks_per_node=walks_per_node,
                    num_neg_samples=num_neg_samples,
                    p=p, 
                    q=q,
                    window=window, 
                    epoch=epoch, 
                    sg=sg, 
                    hs=hs,
                    min_count=min_count,
                    shrink_window=shrink_window)
    
    print(f"embedding size {embed.shape}")
    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}')
    npz_file = os.path.join(root, f'{cfg.data.name}_embeddings.npz')
    if isinstance(embed, dict):
        embeddings_str_keys = {str(key): value for key, value in embed.items()}
        np.savez(npz_file, **embeddings_str_keys)
    else:
        np.savez(npz_file, embeddings=embed)
    
    # embedding method 
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    # dot product
    X_train = embed[X_train_index]
    X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    # dot product 
    X_test = embed[X_test_index]
    X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
    
    
    clf = LogisticRegression(solver='lbfgs', max_iter=iter, multi_class='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)

    acc = clf.score(X_test, y_test)

    plt.figure()
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='test')
    plt.savefig('node2vec_pred.png')
    
    results_acc = {f'{cfg.model.type}_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    result_mrr['ACC'] = acc
    results_mrr = {f'{cfg.model.type}_mrr': result_mrr}
    print(results_acc, results_mrr)
    

    # Save the results
    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}', run_name)
    os.makedirs(root, exist_ok=True)
    acc_file = os.path.join(root, f'{cfg.data.name}_acc.csv')
    mrr_file = os.path.join(root, f'{cfg.data.name}_mrr.csv')
    
    run_id = wandb.util.generate_id()
    append_acc_to_excel(run_id, results_acc, acc_file, cfg.data.name, cfg.model.type)
    append_mrr_to_excel(run_id, results_mrr, mrr_file, cfg.data.name, cfg.model.type)
    
    return result_mrr



# main function 
if __name__ == "__main__":

    args = parse_args()
    # Load args file

    cfg = set_cfg(FILE_PATH, args.cfg)
    seeds = [1, 2, 3, 4, 5]

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    mrr_results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_name = f'seed_{seed}'
        writer = SummaryWriter(log_dir=os.path.join(FILE_PATH, 'runs', cfg.data.name, run_name))
        
        results_mrr = eval_mrr_acc(cfg, writer=writer)
        
        mrr_results.append(results_mrr)
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
    
    # file_path = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}/{cfg.data.name}_model_parameters.csv')
    # with open(file_path, mode='r', newline='') as file:
    #     reader = csv.reader(file)
    #     header = next(reader)

    #     data = []
    #     for row in reader:
    #         data.append([float(value) for value in row[1:]])

    #     rows = np.array(data)

    # means = np.mean(rows, axis=0)
    # mean_row = ['Mean'] + [f'{mean:.6f}' for mean in means]
    # with open(file_path, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(mean_row)