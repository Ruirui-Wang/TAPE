import os
import sys
import os
import sys
import time
from os.path import abspath, dirname, join

import pandas as pd
from torch.nn import BCEWithLogitsLoss
from torch_sparse import SparseTensor


sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from graph_embed.tune_utils import param_tune_acc_mrr
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
from typing import Dict, Tuple
from graphgps.train.opt_train import (Trainer)
from graphgps.utility.ncn import PermIterator

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class Trainer_Embedding(Trainer):
    def __init__(self,
                 FILE_PATH: str,
                 cfg: CN,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data: Data,
                 splits: Dict[str, Data],
                 run: int,
                 repeat: int,
                 loggers: Dict[str, Logger],
                 print_logger: None,
                 batch_size=None,):
        self.device = config_device(cfg).device
        self.model = model.to(self.device)

        self.model_name = cfg.model.model
        self.data_name = cfg.data.name

        self.FILE_PATH = FILE_PATH
        self.epochs = cfg.train.epochs
        self.run = run
        self.repeat = repeat
        self.loggers = loggers
        self.print_logger = print_logger
        self.batch_size = batch_size
        self.data = data

        self.test_data = splits['test']
        self.train_data = splits['train']
        self.valid_data = splits['valid']
        self.optimizer = optimizer
        self.train_func = self._train_embedding
        model_types = ['ComplEx', 'DistMult', 'TransE', 'RotatE']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        self.run = run
        self.repeat = repeat
        self.results_rank = {}

        self.name_tag = cfg.wandb.name_tag
        self.run_result = {}

        self.tensorboard_writer = writer
        self.out_dir = cfg.out_dir
        self.run_dir = cfg.run_dir

        self.report_step = 1

    def _train_embedding(self):
        self.model.train()
        total_loss = 0

        pos_train_tuple = self.train_data['pos_tuple'].to(self.device)
        neg_train_tuple = self.train_data['neg_tuple'].to(self.device)

        # permute the edges
        for perm in PermIterator(pos_train_tuple.device, pos_train_tuple.shape[0], self.batch_size):
            self.optimizer.zero_grad()
            pos_tuple = pos_train_tuple[perm,:]
            neg_tuple = neg_train_tuple[perm,:]
            pos_outs = self.model(pos_tuple) # get the prediction
            neg_outs = self.model(neg_tuple)

            loss = self.model.criterion(pos_outs, neg_outs, torch.tensor([-1]))
            # loss = self.model.criterion(pos_outs, neg_outs)
            
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss

    # def forward(self, pos_sample, neg_sample, mode='single'):
    #     # For the target If :math:`y = 1` then it assumed the first input should be ranked higher
    #     #     (have a larger value) than the second input, and vice-versa for :math:`y = -1`.
    #     target = torch.tensor([-1], dtype=torch.long, device=pos_sample.device)
    #     loss = self.criterion(pos_dist, neg_dist, target)

    #     return loss
    
    
    def train(self):
        best_auc, best_hits10, best_mrr = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_embedding()
            self.tensorboard_writer.add_scalar("Loss/train", loss, epoch)
            if torch.isnan(torch.tensor(loss)):
                print('Loss is nan')
                break
            if epoch % int(self.report_step) == 0:
                self.results_rank = self.merge_result_rank()

                for key, result in self.results_rank.items():
                    self.loggers[key].add_result(self.run, result)
                    self.tensorboard_writer.add_scalar(f"Metrics/Train/{key}", result[0], epoch)
                    self.tensorboard_writer.add_scalar(f"Metrics/Valid/{key}", result[1], epoch)
                    self.tensorboard_writer.add_scalar(f"Metrics/Test/{key}", result[2], epoch)

                    train_hits, valid_hits, test_hits = result
                    self.print_logger.info(
                        f'Run: {self.run + 1:02d}, Key: {key}, '
                        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')

                self.print_logger.info('---')


        return best_auc, best_hits10, best_mrr

    @torch.no_grad()
    def _test(self, data: Data):
        self.model.eval()
        pos_tuple = data['pos_tuple'].to(self.device)
        neg_tuple = data['neg_tuple'].to(self.device)
        pos_pred = torch.cat([self.model(pos_tuple[perm]).squeeze().cpu()
                              for perm in PermIterator(pos_tuple.device, pos_tuple.shape[0], self.batch_size, False)],
                             dim=0)

        neg_pred = torch.cat([self.model(neg_tuple[perm]).squeeze().cpu()
                              for perm in PermIterator(neg_tuple.device, neg_tuple.shape[0], self.batch_size, False)],
                             dim=0)

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_tuple.size(1))
        neg_y = torch.zeros(neg_tuple.size(1))
        y_true = torch.cat([pos_y, neg_y], dim=0)
        '''self.save_pred(y_pred, y_true, data)'''

        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred), auc(fpr, tpr)

    @torch.no_grad()
    def _evaluate(self, eval_data: Data):

        self.model.eval()
        pos_tuple = eval_data['pos_tuple'].to(self.device)
        neg_tuple = eval_data['neg_tuple'].to(self.device)
        pos_pred = torch.cat([self.model(pos_tuple[perm]).squeeze().cpu()
                              for perm in PermIterator(pos_tuple.device, pos_tuple.shape[0], self.batch_size, False)],
                             dim=0)

        neg_pred = torch.cat([self.model(neg_tuple[perm]).squeeze().cpu()
                              for perm in PermIterator(neg_tuple.device, neg_tuple.shape[0], self.batch_size, False)],
                             dim=0)

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_tuple.size(0))
        neg_y = torch.zeros(neg_tuple.size(0))
        y_true = torch.cat([pos_y, neg_y], dim=0)
        '''self.save_pred(y_pred, y_true, data)'''

        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        acc = (y_pred == y_true).sum() / len(y_true)

        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})

        return result_mrr




