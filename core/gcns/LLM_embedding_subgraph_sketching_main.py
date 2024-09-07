import copy
import itertools
import os, sys

from torch_geometric.typing import torch_sparse
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.nn.conv.gcn_conv import gcn_norm


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphgps.network.subgraph_sketching import ElphHashes

import argparse
import time
import logging
import wandb
import torch
from functools import partial
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
          create_optimizer, config_device,  create_logger, custom_set_out_dir, save_run_results_to_csv

import scipy.sparse as ssp
from graphgps.network.subgraph_sketching import BUDDY, ELPH

from data_utils.load import load_data_lp
from graphgps.train.subgraph_sketching_train import Trainer_Subgraph_Sketching

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncnc.yaml',
                        help='The configuration file path.')

    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--embedder', type=str, required=False, default='llama')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--device', dest='device', required=True,
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=400,
                        help='data name')
    parser.add_argument('--wandb', dest='wandb', required=False,
                        help='data name')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()
def _generate_sign_features(data, edge_index, edge_weight, sign_k):
    """
    Generate features by preprocessing using the Scalable Inception Graph Neural Networks (SIGN) method
     https://arxiv.org/abs/2004.11198
    @param data: A pyg data object
    @param sign_k: the maximum number of times to apply the propagation operator
    @return:
    """
    try:
        num_nodes = data.x.size(0)
    except AttributeError:
        num_nodes = data.num_nodes
    edge_index, edge_weight = gcn_norm(  # yapf: disable
        edge_index, edge_weight.float(), num_nodes)
    if sign_k == 0:
        # for most datasets it works best do one step of propagation
        xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
    else:
        xs = [data.x]
        for _ in range(sign_k):
            x = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
            xs.append(x)
        xs = torch.cat(xs, dim=-1)
    return xs


def _preprocess_subgraph_features(num_nodes, edge_index, edges):
    """
    Handles caching of hashes and subgraph features where each edge is fully hydrated as a preprocessing step
    Sets self.subgraph_features
    @return:
    """
    elph_hashes = ElphHashes(cfg.model)
    hashes, cards = elph_hashes.build_hash_tables(num_nodes, edge_index)
    subgraph_features = elph_hashes.get_subgraph_features(edges, hashes, cards, cfg.train.batch_size)
    if cfg.model.floor_sf and subgraph_features is not None:
        subgraph_features[subgraph_features < 0] = 0
    if not cfg.model.use_zero_one and subgraph_features is not None:  # knock out the zero_one features (0,1) and (1,0)
        if cfg.model.max_hash_hops > 1:
            subgraph_features[:, [4, 5]] = 0
        if cfg.model.max_hash_hops == 3:
            subgraph_features[:, [11, 12]] = 0  # also need to get rid of (0, 2) and (2, 0)
    return subgraph_features
def hash_dataset(splits):
    data = splits.cpu()
    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    edge_index = data.edge_index.cpu()
    edge_weight = edge_weight.cpu()
    data.links = torch.cat([data['pos_edge_label_index'], data['neg_edge_label_index']], dim=1)
    data.A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])),
                            shape=(data.num_nodes, data.num_nodes))
    data.x = _generate_sign_features(data, edge_index, edge_weight, cfg.model.sign_k)
    data.subgraph_features = _preprocess_subgraph_features(data.num_nodes, data.edge_index, data.links.T)
    data.degrees = torch.tensor(data.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()
    return data


if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data

    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]

    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    cfg.device = args.device
    cfg.wandb.name_tag = args.embedder + cfg.wandb.name_tag

    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)
        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        start = time.time()
        splits, __, data = load_data_lp[cfg.data.name](cfg.data)
        saved_features_path = './' + args.embedder + cfg.data.name + 'saved_node_features.pt'
        if os.path.exists(saved_features_path):
            node_features = torch.load(saved_features_path, map_location=torch.device('cpu'))
            data.x = node_features
            data.x = torch.tensor(data.x)
            splits['train'].x = torch.tensor(data.x)
            splits['valid'].x = torch.tensor(data.x)
            splits['test'].x = torch.tensor(data.x)
            print('Replaced node_features with saved features')
        else:
            print('Please regenerate node features')

        data.edge_index = torch.cat([splits['train']['pos_edge_label_index'],
                                     splits['train']['neg_edge_label_index']], dim=1)
        if cfg.model.type == 'BUDDY':
            splits['train'] = hash_dataset(splits['train'])
            splits['valid'] = hash_dataset(splits['valid'])
            splits['test'] = hash_dataset(splits['test'])
            model = BUDDY(cfg.model, data.num_features, node_embedding=None)
        elif cfg.model.type == 'ELPH':
            model = ELPH(cfg.model, data.num_features, node_embedding=None)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.lr,
                                         weight_decay=cfg.optimizer.weight_decay)

        # Execute experiment
        trainer = Trainer_Subgraph_Sketching(FILE_PATH,
                                             cfg,
                                             model,
                                             optimizer,
                                             data,
                                             splits,
                                             run_id,
                                             args.repeat,
                                             loggers,
                                             print_logger=print_logger,
                                             batch_size=cfg.train.batch_size)

        start = time.time()
        trainer.train()
        end = time.time()
        print('Training time: ', end - start)
        save_run_results_to_csv(cfg, loggers, seed, run_id)


    print('All runs:')

    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
        result_dict[key] = valid_test

    trainer.save_result(result_dict)

    cfg.model.params = params_count(model)
    print_logger.info(f'Num parameters: {cfg.model.params}')
    trainer.finalize()
    print_logger.info(f"Inference time: {trainer.run_result['eval_time']}")
