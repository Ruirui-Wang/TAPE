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


class Trainer_HLGNN(Trainer):
    def __init__(self,
                 FILE_PATH: str,
                 cfg: CN,
                 model: torch.nn.Module,
                 predictor: torch.nn.Module,
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
        self.predictor = predictor.to(self.device)

        self.model_name = cfg.model.type
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
        self.train_func = self._train_hlgnn
        model_types = ['HLGNN']
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

    def _train_hlgnn(self):
        self.model.train()
        total_loss = 0
        self.predictor.train()

        pos_train_edge = self.train_data['pos_edge_label_index'].to(self.device)
        adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool) # mask for adj
        negedge = self.train_data['neg_edge_label_index'].to(self.device)
        # permute the edges
        for perm in PermIterator(adjmask.device, adjmask.shape[0], self.batch_size):
            self.optimizer.zero_grad()
            # mask input edges (target link removal)
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask] # get the target edge index
            # get the adj matrix
            adj = SparseTensor.from_edge_index(tei, sparse_sizes=(self.data.num_nodes, self.data.num_nodes)).to_device(
                pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
            
            h = self.model(self.data.x, self.data.adj_t, self.data.edge_weight) # get the node embeddings
            edge = pos_train_edge[:, perm]
            pos_outs = self.predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_outs + 1e-15).mean()
            edge = negedge[:, perm]
            neg_outs = self.predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_outs + 1e-15).mean()
            loss = neg_loss + pos_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss

    def train(self):
        best_auc, best_hits10, best_mrr = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_hlgnn()
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
        self.predictor.eval()
        pos_edge = data['pos_edge_label_index'].to(self.device)
        neg_edge = data['neg_edge_label_index'].to(self.device)
        if data == self.test_data:
            adj = self.data.full_adj_t
            h = self.model(self.data.x, self.data.adj_t, self.data.edge_weight)
        else:
            adj = self.data.adj_t
            h = self.model(self.data.x, self.data.adj_t, self.data.edge_weight)
        pos_pred = torch.cat([self.predictor(h[pos_edge[0]], h[pos_edge[1]]).squeeze().cpu()
                              for perm in PermIterator(pos_edge.device, pos_edge.shape[0], self.batch_size, False)],
                             dim=0)

        neg_pred = torch.cat([self.predictor(h[neg_edge[0]], h[neg_edge[1]]).squeeze().cpu()
                              for perm in PermIterator(neg_edge.device, neg_edge.shape[0], self.batch_size, False)],
                             dim=0)

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_edge.size(1))
        neg_y = torch.zeros(neg_edge.size(1))
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
        self.predictor.eval()
        pos_edge = eval_data['pos_edge_label_index'].to(self.device)
        neg_edge = eval_data['neg_edge_label_index'].to(self.device)
        if eval_data == self.test_data:
            adj = self.data.full_adj_t
            h = self.model(self.data.x, self.data.adj_t, self.data.edge_weight)
        else:
            adj = self.data.adj_t
            h = self.model(self.data.x, self.data.adj_t, self.data.edge_weight)
        pos_pred = torch.cat([self.predictor(h[pos_edge[0]], h[pos_edge[1]]).squeeze().cpu()
                              for perm in PermIterator(pos_edge.device, pos_edge.shape[0], self.batch_size, False)],
                             dim=0)

        neg_pred = torch.cat([self.predictor(h[neg_edge[0]], h[neg_edge[1]]).squeeze().cpu()
                              for perm in PermIterator(neg_edge.device, neg_edge.shape[0], self.batch_size, False)],
                             dim=0)

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_edge.size(1))
        neg_y = torch.zeros(neg_edge.size(1))
        y_true = torch.cat([pos_y, neg_y], dim=0)
        '''self.save_pred(y_pred, y_true, eval_data)'''

        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()
        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        acc = torch.sum(y_true == y_pred) / len(y_true)

        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})

        return result_mrr


    def finalize(self):
        eval_data = self.test_data
        start_train = time.time()

        self.model.eval()
        self.predictor.eval()
        pos_edge = eval_data['pos_edge_label_index'].to(self.device)
        neg_edge = eval_data['neg_edge_label_index'].to(self.device)
        if eval_data == self.test_data:
            adj = self.data.full_adj_t
            h = model(self.data.x, self.data.adj_t, self.data.edge_weight)
        else:
            adj = self.data.adj_t
            h = model(self.data.x, self.data.adj_t, self.data.edge_weight)
        pos_edge_indices = []
        neg_edge_indices = []
        pos_pred = []
        neg_pred = []

        for perm in PermIterator(pos_edge.device, pos_edge.shape[1], self.batch_size, False):
            pos_pred_batch = self.predictor(h, adj, pos_edge[:, perm]).squeeze().cpu()
            pos_pred.append(pos_pred_batch)  # Append to list
            pos_edge_indices.append(pos_edge[:, perm].cpu())

        for perm in PermIterator(neg_edge.device, neg_edge.shape[1], self.batch_size, False):
            neg_pred_batch = self.predictor(h, adj, neg_edge[:,perm]).squeeze().cpu()
            neg_pred.append(neg_pred_batch)  # Append to list
            neg_edge_indices.append(neg_edge[:, perm].cpu())

        pos_pred = torch.cat(pos_pred, dim=0)
        neg_pred = torch.cat(neg_pred, dim=0)


        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_pred = pos_pred.detach().cpu()
        neg_pred = neg_pred.detach().cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        edge_index = torch.cat([pos_edge, neg_edge],dim=1)
        pos_y = torch.ones(pos_edge.size(1))
        neg_y = torch.zeros(neg_edge.size(1))
        y_true = torch.cat([pos_y, neg_y], dim=0)
        data_df = {
            "edge_index0": edge_index[0].detach().cpu().numpy(),
            "edge_index1": edge_index[1].detach().cpu().numpy(),
            "pred": y_pred.detach().cpu().numpy(),
            "gr": y_true.detach().cpu().numpy(),
        }

        df = pd.DataFrame(data_df)
        auc = result_mrr['AUC']
        mrr = result_mrr['MRR']
        df.to_csv(f'{self.out_dir}/{self.name_tag}_AUC_{auc}_MRR_{mrr}.csv', index=False)
        self.run_result['eval_time'] = time.time() - start_train
        return



