import copy
import itertools
import os, sys

import transformers
import wandb
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
    create_optimizer, config_device, \
    create_logger, custom_set_out_dir
from torch_geometric.graphgym.utils.comp_budget import params_count
from data_utils.load import load_data_lp, load_graph_lp
from graphgps.train.embedding_LLM_train import Trainer_embedding_LLM
from graphgps.utility.utils import save_run_results_to_csv
from graphgps.config import dump_run_cfg


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncnc.yaml',
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    parser.add_argument('--device', dest='device', required=True,
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=300,
                        help='data name')
    parser.add_argument('--wandb', dest='wandb', required=False,
                        help='data name')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

if __name__ == '__main__':
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch

    torch.set_num_threads(cfg.num_threads)
    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    cfg.device = args.device
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)


    splits, text, data = load_data_lp[cfg.data.name](cfg.data)
    if cfg.embedder.type == 'minilm':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=cfg.device)
        node_features = model.encode(text, batch_size=256)
    elif cfg.embedder.type == 'e5-large':
        model = SentenceTransformer('intfloat/e5-large-v2', device=cfg.device)
        node_features = model.encode(text, normalize_embeddings=True, batch_size=256)
    elif cfg.embedder.type == 'llama':
        model_id = "meta-llama/Meta-Llama-3-8B"
        pipeline = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        node_features = pipeline(text)
    elif cfg.embedder.type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased").to(cfg.device)
        node_features = []
        batch_size = 256  # Adjust batch size as needed to avoid OOM errors
        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i + batch_size]
            encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                      max_length=512).to(cfg.device)
            with torch.no_grad():
                outputs = model(**encoded_input)
                batch_features = outputs.pooler_output
                node_features.append(batch_features)
        node_features = torch.cat(node_features, dim=0)
    node_features = torch.tensor(node_features)
    print(node_features.shape)

    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)

        hyperparameter_search = {'base_lr': [0.01, 0.001, 0.0001], 'batch_size': [64, 128, 256, 512, 1024, 2048]}
        print_logger.info(f"hypersearch space: {hyperparameter_search}")
        for base_lr, batch_size in tqdm(itertools.product(*hyperparameter_search.values())):
            cfg.optimizer.base_lr = base_lr
            cfg.train.batch_size = batch_size


            model = LinkPredictor(node_features.shape[1], cfg.model.hidden_channels, 1, cfg.model.num_layers, cfg.model.dropout)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.base_lr, weight_decay=cfg.optimizer.weight_decay)
            logging.info(f"{model} on {next(model.parameters()).device}")
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')

            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
            custom_set_run_dir(cfg, cfg.wandb.name_tag)

            dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {run_id} with seed {seed} on device {cfg.device}')
            trainer = Trainer_embedding_LLM(FILE_PATH,
                                                 cfg,
                                                 model,
                                                 optimizer,
                                                 node_features,
                                                 splits,
                                                 run_id,
                                                 args.repeat,
                                                 loggers,
                                                 print_logger=print_logger,
                                                 batch_size=cfg.train.batch_size)

            start = time.time()
            trainer.train()
            end = time.time()
            run_result = {}
            for key in trainer.loggers.keys():
                # refer to calc_run_stats in Logger class
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                run_result[key] = test_bvalid

            run_result.update(
                {"base_lr": cfg.optimizer.base_lr, "batch_size": cfg.train.batch_size})
            print_logger.info(run_result)

            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)




