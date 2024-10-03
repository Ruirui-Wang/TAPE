import copy
import os, sys

from torch_sparse import SparseTensor

from torch_geometric.graphgym import params_count

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
from functools import partial

from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
    create_optimizer, config_device, \
    create_logger, save_run_results_to_csv

import scipy.sparse as ssp

from graphgps.network.neognn import NeoGNN, LinkPredictor
from data_utils.load import load_data_lp
from graphgps.train.neognn_train import Trainer_NeoGNN
from graphgps.network.ncn import predictor_dict



def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')

    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
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

def ngnn_dataset(splits):
    for data in splits.values():
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
        data.edge_weight = None
        data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.emb = data.x
        edge_weight = torch.ones(edge_index.size(1), dtype=float)
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        data.A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])),
                           shape=(data.num_nodes, data.num_nodes))
        A2 = data.A * data.A
        data.A = data.A + cfg.model.beta * A2
    return splits


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

    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)
        splits = ngnn_dataset(splits)
        path = f'{os.path.dirname(__file__)}/neognn_{cfg.data.name}'
        dataset = {}

        model = NeoGNN(data.x.shape[1], cfg.model.hidden_channels,
                       cfg.model.hidden_channels, cfg.model.num_layers,
                       cfg.model.dropout, args=cfg.model)

        predictor = LinkPredictor(cfg.model.hidden_channels, cfg.model.hidden_channels, 1,
                                  cfg.model.mlp_num_layers, cfg.model.dropout)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

        # Execute experiment
        trainer = Trainer_NeoGNN(FILE_PATH,
                               cfg,
                               model,
                               predictor,
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
