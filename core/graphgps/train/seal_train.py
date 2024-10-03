import os
import sys
import os
import sys
from os.path import abspath, dirname, join

from torch.nn import BCEWithLogitsLoss


sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import pandas as pd 
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN
from graphgps.utility.utils import Logger, config_device

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from heuristic.eval import get_metric_score
from graphgps.train.opt_train import Trainer
from graphgps.train.heart_train import Trainer_Heart
from typing import Any, List, Optional, Sequence, Union, Dict
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class Trainer_SEAL(Trainer):
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
                 batch_size=None, ):

        self.device = config_device(cfg).device
        self.model = model.to(self.device)

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
        self.train_func = self._train_seal
        model_types = ['SEAL']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

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
        
    def _train_seal(self):
        self.model.train()
        total_loss = 0
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for subgraph in train_loader:
            data = subgraph.to(self.device)
            self.optimizer.zero_grad()
            x = subgraph.x
            edge_weight = subgraph.edge_weight
            node_id = subgraph.node_id
            logits = self.model(subgraph.z, subgraph.edge_index, subgraph.batch, x, edge_weight, node_id)
            loss = BCEWithLogitsLoss()(logits.view(-1), subgraph.y.to(torch.float))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.train_data)
    

    def train(self):  
        for epoch in range(1, self.epochs + 1):
            loss = self._train_seal()
        
            self.tensorboard_writer.add_scalar("Loss/train", loss, epoch)

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

                
    @torch.no_grad()
    def _test(self, data: Data):
        self.model.eval()
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        y_pred, y_true = [], []
        for data in test_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = data.edge_weight
            node_id = data.node_id
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred), auc(fpr, tpr)


    @torch.no_grad()
    def _evaluate(self, eval_data: Data):

        self.model.eval()
        data_loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False)
        y_pred, y_true = [], []
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = data.edge_weight
            node_id = data.node_id
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        y_pred, y_true = y_pred.cpu(), y_true.cpu()

        hard_thres = (y_pred.max() + y_pred.min()) / 2

        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()
        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        acc = torch.sum(y_true == y_pred) / len(y_true)

        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})

        return result_mrr

    def finalize(self):
        import time 
        for _ in range(1):
            start_train = time.time() 
            self._evaluate(self.test_data)
            self.run_result['eval_time'] = time.time() - start_train


    @torch.no_grad()
    def save_eval_edge_pred(self, eval_data: Data):    
        self.model.eval()
        y_pred, y_true, edge_index_s, edge_index_t = [], [], [], []

        for data in DataLoader(eval_data, batch_size=self.batch_size, shuffle=False):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = data.edge_weight
            node_id = data.node_id
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.extend(logits.view(-1).cpu().tolist())
            y_true.extend(data.y.view(-1).cpu().to(torch.float).tolist())
            edge_index_s.extend(np.asarray(data.st).T[0]) 
            edge_index_t.extend(np.asarray(data.st).T[1]) 

        data_df = {
            "edge_index0": edge_index_s,
            "edge_index1": edge_index_t,
            "pred": y_pred,
            "gr": y_true,
        }
        
        df = pd.DataFrame(data_df)
        df.to_csv(f'{self.run_dir}/{self.data_name}_test_pred_gr_last_epoch.csv', index=False)
        return

'''    def final_evaluate(self, eval_data: Data):
        self.model.eval()
        data_loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False)
        y_pred, y_true = [], []
        edge_index = []
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = data.edge_weight
            node_id = data.node_id
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))
            edge_index.append(data.)

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        y_pred, y_true = y_pred.cpu(), y_true.cpu()

        data_df = {
            "edge_index0": edge_index_s,
            "edge_index1": edge_index_t,
            "pred": y_pred,
            "gr": y_true,
        }

        df = pd.DataFrame(data_df)
        df.to_csv(f'{self.run_dir}/{self.data_name}_test_pred_gr_last_epoch.csv', index=False)
        return
'''
