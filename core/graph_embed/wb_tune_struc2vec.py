import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import scipy.sparse as ssp
from sklearn.linear_model import LogisticRegression
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_scipy_sparse_matrix
from ogb.linkproppred import Evaluator
from yacs.config import CfgNode as CN

from heuristic.eval import get_metric_score
from data_utils.load import load_data_lp as data_loader

from core.graphgps.utility.utils import (
    get_git_repo_root_path
)

from ge.models import Struc2Vec
import wandb
from core.graph_embed.tune_utils import (
    set_cfg,
    parse_args,
    load_sweep_config, 
    initialize_config, 
    param_tune_acc_mrr,
    FILE_PATH,
)


if __name__ == "__main__":

    args = parse_args()
    print(args)

    SWEEP_FILE_PATH = FILE_PATH + args.sweep_file
    sweep_config = load_sweep_config(SWEEP_FILE_PATH)

    cfg = initialize_config(args)

    dataset, _, splits = data_loader[cfg.data.name](cfg)

    FILE_PATH = get_git_repo_root_path() + '/'

    cfg_file = FILE_PATH + args.cfg_file
    
    with open(cfg_file, "r") as f:
        cfg = CN.load_cfg(f)
    
    # Access individual parameters
    max_iter = cfg.model.struc2vec.max_iter
    
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    result_dict = {}
    
    adj = to_scipy_sparse_matrix(full_edge_index)

    G = nx.from_scipy_sparse_array(adj)
    
    import random 
    # three parameters
    for i in range(40):
        tune_dict = sweep_config['parameters']
        for wl in tune_dict['wl']['values']:
            for nw in tune_dict['num_walks']['values']:
                for es in tune_dict['embed_size']['values']:
                    for ws in tune_dict['window_size']['values']:
        
                        print(wl, nw, es, ws)
                        model = Struc2Vec(G, 
                                            walk_length= wl, 
                                            num_walks = nw, 
                                            workers=20, 
                                            verbose=40, 
                                            data=cfg.data.name, 
                                            reuse=False, 
                                            temp_path=f'./temp_path')
                        
                        model.train(embed_size=es, 
                                    window_size=ws, 
                                    workers=20)

                        embed = model.get_embeddings()

                        print(f"embedding size {embed.shape}")

                        # embedding method 
                        X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
                        # dot product
                        X_train = embed[X_train_index]
                        X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
                        X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
                        # dot product 
                        X_test = embed[X_test_index]
                        X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
                    

                        clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='auto')
                        clf.fit(X_train, y_train)
                        
                        y_pred = clf.predict_proba(X_test)

                        acc = clf.score(X_test, y_test)

                        plt.figure()
                        plt.plot(y_pred, label='pred')
                        plt.plot(y_test, label='test')
                        plt.savefig(f'ws{ws}wl{wl}es{es}ws{ws}struc2vec_pred_{cfg.data.name}.png')
                        
                        results_acc = {'node2vec_acc': acc, 'wl': wl, 'nw': nw, 'es': es, 'ws': ws}
                        
                        pos_test_pred = torch.tensor(y_pred[y_test == 1])
                        neg_test_pred = torch.tensor(y_pred[y_test == 0])
                        
                        evaluator_hit = Evaluator(name='ogbl-collab')
                        evaluator_mrr = Evaluator(name='ogbl-citation2')
                        pos_pred = pos_test_pred[:, 1]
                        neg_pred = neg_test_pred[:, 1] 
                        result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
                        results_acc.update(result_mrr)


                        root = FILE_PATH + 'results'
                        acc_file = root + f'/{cfg.data.name}_acc_struc2vec.csv'

                        if not os.path.exists(root):
                            os.makedirs(root, exist_ok=True)
                        
                        id = wandb.util.generate_id()
                        param_tune_acc_mrr(id, results_acc, acc_file, cfg.data.name, 'struc2vec')
                        


