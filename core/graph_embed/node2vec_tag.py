import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

import numpy as np
import scipy.sparse as ssp
import torch
import matplotlib.pyplot as plt
from core.data_utils.graph_stats import plot_coo_matrix, construct_sparse_adj
from ogb.linkproppred import Evaluator
from heuristic.eval import get_metric_score
from graph_embed.ge.models import Node2Vec
from yacs.config import CfgNode as CN
import networkx as nx 
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg)
from graph_embed.node2vec_tagplus import node2vec
from data_utils.load import load_data_lp as data_loader

from core.graphgps.utility.utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel

FILE_PATH = get_git_repo_root_path() + '/'


def set_cfg(args):
    with open(args.cfg_file, "r") as f:
        cfg = CN.load_cfg(f)
    return cfg


        
def eval_embed(embed,  splits, visual=True):
    """train the classifier and return the pred

    Args:
        embed (np.array): embedding vector generated from the graph
        test_index (torch.tensor): test edge index
        edge_label (torch.tensor): test edge label

    Returns:
        prediction for edges 
    """
    # train loop
    embed = np.asarray(list(embed.values()))
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    # dot product
    X_train = embed[X_train_index]
    X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    # dot product 
    X_test = embed[X_test_index]
    X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
    
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(100,), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001, 
                        batch_size='auto', 
                        learning_rate_init=0.001, 
                        power_t=0.5, 
                        max_iter=200, 
                        shuffle=True, 
                        random_state=None, 
                        tol=0.0001, 
                        verbose=False, 
                        warm_start=False, 
                        momentum=0.9, 
                        nesterovs_momentum=True, 
                        early_stopping=False, 
                        validation_fraction=0.1, 
                        epsilon=1e-08, 
                        n_iter_no_change=10, 
                        max_fun=15000).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = clf.score(X_test, y_test)
    
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    results_mrr = {'node2vec_mrr': result_mrr}
    
    if visual:
        model = TSNE(n_components=2,
                    init="random",
                    random_state=0,
                    perplexity=100,
                    n_iter=300)
        node_pos = model.fit_transform(X_test)

        color_dict = {'0.0': 'r', '1.0': 'b'}
        color = [color_dict[str(i)] for i in y_test.tolist()]
        plt.figure()
        for idx in range(len(node_pos)):
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c=color[idx])
        plt.legend()
        plt.savefig(f'cora_node2vec.png')
    return y_pred, results_acc, results_mrr, y_test 


def eval_pubmed_mrr_acc(args) -> None:
    """load text attribute graph in link predicton setting
    """
    dataset, data_cited, splits = data_loader[args.data.name](args)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
        
    result_dict = {}
    # Access individual parameters
    walk_length = args.model.node2vec.walk_length
    num_walks = args.model.node2vec.num_walks
    p = args.model.node2vec.p
    q = args.model.node2vec.q
    workers = args.model.node2vec.workers
    use_rejection_sampling = args.model.node2vec.use_rejection_sampling
    embed_size = args.model.node2vec.embed_size
    ws = args.model.node2vec.window_size
    iter = args.model.node2vec.iter

    # model 
    for use_heuristic in ['node2vec']:
        G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
        model = Node2Vec(G, walk_length=walk_length, 
                         num_walks=num_walks,
                        p=p, 
                        q=q, 
                        workers=workers, 
                        use_rejection_sampling=use_rejection_sampling)
        
        model.train(embed_size=embed_size,
                    window_size = ws, 
                    iter = iter)
        
        embeddings = model.get_embeddings()


    return eval_embed(embeddings, splits)

from sklearn.manifold import TSNE
from heuristic.pubmed_heuristic import get_pubmed_casestudy   




# main function 
if __name__ == "__main__":
    args = parse_args()
    # Load args file
    
    cfg = set_cfg(args)
    cfg.merge_from_list(args.opts)


    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    y_pred, results_acc, results_mrr, y_test  = eval_pubmed_mrr_acc(cfg)

    root = FILE_PATH + 'results'
    acc_file = root + f'/{cfg.data.name}_acc.csv'
    mrr_file = root +  f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    append_acc_to_excel(results_acc, acc_file, cfg.data.name)
    append_mrr_to_excel(results_mrr, mrr_file)

