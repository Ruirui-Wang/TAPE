import itertools
import os, sys

from torch_sparse import SparseTensor
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import logging
import wandb
import torch
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
          create_optimizer, config_device,  create_logger, custom_set_out_dir
import scipy.sparse as ssp
from graphgps.config import (dump_cfg, dump_run_cfg)
from graphgps.network.neognn import NeoGNN, LinkPredictor

from data_utils.load import load_data_lp
from graphgps.train.neognn_train import Trainer_NeoGNN


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/seal.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='cora',
                        help='data name')
    parser.add_argument('--device', dest='device', required=True,
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=50,
                        help='data name')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
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
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]

    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)


    for run_id, seed, split_index in zip(
                *run_loop_settings(cfg, args)):
        id = wandb.util.generate_id()
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)

        path = f'{os.path.dirname(__file__)}/neognn_{cfg.data.name}'
        print_logger = set_printing(cfg)
        print_logger.info(
            f"The {cfg['data']['name']} graph {splits['train']['x'].shape} is loaded on {splits['train']['x'].device},"
            f"\n Train: {2 * splits['train']['pos_edge_label'].shape[0]} samples,"
            f"\n Valid: {2 * splits['train']['pos_edge_label'].shape[0]} samples,"
            f"\n Test: {2 * splits['test']['pos_edge_label'].shape[0]} samples")
        dump_cfg(cfg)
        hyperparameter_search = {'hidden_channels': [128, 256], 'num_layers': [1, 2, 3], 'mlp_num_layers': [2],
                                 'f_edge_dim': [8], 'f_node_dim': [128], 'dropout': [0, 0.1, 0.3],
                             "batch_size": [256, 512, 1024], "lr": [0.01, 0.001, 0.0001]}

        print_logger.info(f"hypersearch space: {hyperparameter_search}")
        for hidden_channels, num_layers, mlp_num_layers, f_edge_dim, f_node_dim, dropout, batch_size, lr in tqdm(
                itertools.product(*hyperparameter_search.values())):
            cfg.model.hidden_channels = hidden_channels
            cfg.train.batch_size = batch_size
            cfg.optimizer.lr = lr
            cfg.model.num_layers = num_layers
            cfg.model.mlp_num_layers = mlp_num_layers
            cfg.model.f_node_dim = f_node_dim
            cfg.model.f_edge_dim = f_edge_dim
            cfg.model.dropout = dropout
            splits = ngnn_dataset(splits)

            print_logger.info(
                f"hidden_channels: {hidden_channels}, num_layers: {num_layers}, mlp_num_layers:{mlp_num_layers}, f_node_dim: {f_node_dim}, "
                f"f_edge_dim: {f_edge_dim}, dropout: {dropout}, batch_size: {batch_size}, lr: {lr}")
            start_time = time.time()
            model = NeoGNN(data.x.shape[1], cfg.model.hidden_channels,
                           cfg.model.hidden_channels, cfg.model.num_layers,
                           cfg.model.dropout, args=cfg.model)

            predictor = LinkPredictor(cfg.model.hidden_channels, cfg.model.hidden_channels, 1,
                                      cfg.model.mlp_num_layers, cfg.model.dropout)

            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(predictor.parameters()), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

            logging.info(f"{model} on {next(model.parameters()).device}")
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')

            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
            custom_set_run_dir(cfg, cfg.wandb.name_tag)

            dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')

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
                                  batch_size=batch_size)

            trainer.train()

            run_result = {}
            for key in trainer.loggers.keys():
                if trainer.loggers[key].results == [[], []]:
                    run_result[key] = None
                else:
                # refer to calc_run_stats in Logger class
                    _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                    run_result[key] = test_bvalid

            run_result.update(
                {'hidden_channels': hidden_channels,' num_layers': num_layers, 'mlp_num_layers': mlp_num_layers,
                 'f_node_dim': f_node_dim, 'f_edge_dim': f_edge_dim, 'dropout': dropout, 'batch_size': batch_size, 'lr': lr})
            print_logger.info(run_result)

            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)

            print_logger.info(f"runing time {time.time() - start_time}")
            torch.cuda.empty_cache()
