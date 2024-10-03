import os
import sys
import time
from os.path import abspath, dirname, join

import numpy as np
from torch_scatter import scatter_add

sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
from typing import Dict, Tuple
from graphgps.train.opt_train import (Trainer)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class Trainer_NeoGNN(Trainer):
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
        self.gnn_batch_size = cfg.train.gnn_batch_size
        self.data = data.to(self.device)

        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        self.train_func = self._train_neognn
        model_types = ['NeoGNN']
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

    def _train_neognn(self):
        self.model.train()
        total_loss = 0
        self.predictor.train()
        pos_train_edge = self.train_data['pos_edge_label_index'].to(self.device).T
        neg_train_edge = self.train_data['neg_edge_label_index'].to(self.device).T
        # permute the edges
        total_loss = total_examples = 0
        count = 0
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True):
            self.optimizer.zero_grad()
            # compute scores of positive edges
            edge = pos_train_edge[perm].t()
            pos_out, pos_out_struct, _, _ = self.model(edge, self.train_data, self.train_data.A, self.predictor,
                                                       emb=self.train_data.emb)
            _, _, pos_out_feat_large = self.model(edge, self.train_data, self.train_data.A, self.predictor,
                                                  emb=self.train_data.emb, only_feature=True)

            # compute scores of negative edges
            # Just do some trivial random sampling.
            edge = neg_train_edge[perm].t()
            neg_out, neg_out_struct, _, _ = self.model(edge, self.train_data, self.train_data.A, self.predictor,
                                                       emb=self.train_data.emb)
            _, _, neg_out_feat_large = self.model(edge, self.train_data, self.train_data.A, self.predictor,
                                                  emb=self.train_data.emb, only_feature=True)

            if pos_out_struct != None:
                pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
                neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
                loss1 = pos_loss + neg_loss
            else:
                loss1 = 0
            if pos_out_feat_large != None:
                pos_loss = -torch.log(pos_out_feat_large + 1e-15).mean()
                neg_loss = -torch.log(1 - neg_out_feat_large + 1e-15).mean()
                loss2 = pos_loss + neg_loss
            else:
                loss2 = 0
            if pos_out != None:
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
                loss3 = pos_loss + neg_loss
            else:
                loss3 = 0
            loss = loss1 + loss2 + loss3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            count += 1

        return total_loss / total_examples
    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_neognn()
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

        return best_auc, best_hits


    @torch.no_grad()
    def _evaluate(self, eval_data: Data):
        self.model.eval()
        self.predictor.eval()

        h = self.model.forward_feature(eval_data.x, eval_data.adj_t)

        pos_edge = eval_data['pos_edge_label_index'].to(self.device)
        neg_edge = eval_data['neg_edge_label_index'].to(self.device)

        pos_pred_list, neg_pred_list = [], []

        edge_weight = torch.from_numpy(eval_data.A.data).to(h.device)
        edge_weight = self.model.f_edge(edge_weight.unsqueeze(-1))

        row, col = eval_data.A.nonzero()
        edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).type(torch.LongTensor).to(h.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=self.data.num_nodes)
        deg = self.model.f_node(deg).squeeze()
        deg = deg.cpu().numpy()
        A_ = eval_data.A.multiply(deg).tocsr()

        alpha = torch.softmax(self.model.alpha, dim=0).cpu()
        print(alpha)

        with torch.no_grad():
            for perm in DataLoader(range(pos_edge.size(1)), self.batch_size, shuffle=True):
                # Positive edge prediction
                edge = pos_edge[:, perm]
                gnn_scores = self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
                src, dst = edge.cpu()
                cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
                cur_scores = torch.sigmoid(self.model.g_phi(cur_scores).squeeze().cpu())
                cur_scores = alpha[0] * cur_scores + alpha[1] * gnn_scores
                pos_pred_list += [cur_scores]

                # Negative edge prediction
                edge = neg_edge[:, perm]
                gnn_scores = self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
                src, dst = edge.cpu()
                cur_scores = torch.from_numpy(np.sum(A_[src].multiply(A_[dst]), 1)).to(h.device)
                cur_scores = torch.sigmoid(self.model.g_phi(cur_scores).squeeze().cpu())
                cur_scores = alpha[0] * cur_scores + alpha[1] * gnn_scores
                neg_pred_list += [cur_scores]

        # Concatenate predictions and create labels
        pos_pred = torch.cat(pos_pred_list, dim=0)
        neg_pred = torch.cat(neg_pred_list, dim=0)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)

        hard_thres = (y_pred.max() + y_pred.min()) / 2

        # Create true labels
        pos_y = torch.ones(pos_pred.size(0))
        neg_y = torch.zeros(neg_pred.size(0))
        y_true = torch.cat([pos_y, neg_y], dim=0)

        # Convert predictions to binary labels
        y_pred_binary = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        # Move to CPU for evaluation
        y_true = y_true.cpu()
        y_pred_binary = y_pred_binary.cpu()

        acc = torch.sum(y_true == y_pred_binary).item() / len(y_true)

        # Assuming get_metric_score is defined elsewhere and computes metrics
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc, 5)})

        return result_mrr

    def save_pred(self, pred, true, data):
        root = os.path.join(self.FILE_PATH, cfg.out_dir, 'pred_record')
        os.makedirs(root, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        file_path = os.path.join(root, f'{cfg.dataset.name}_{timestamp}.txt')

        with open(file_path, 'w') as f:
            for idx, subgraph in enumerate(data):
                indices = torch.where(subgraph['z'] == 1)[0]
                if len(indices) < 2:
                    continue
                corresponding_node_ids = subgraph['node_id'][indices]
                pred_value = pred[idx]
                true_value = true[idx]
                f.write(f"{corresponding_node_ids[0].item()} {corresponding_node_ids[1].item()} {pred_value} {true_value}\n")
    def finalize(self):
        import time
        for _ in range(1):
            start_train = time.time()
            self._evaluate(self.test_data)
            self.run_result['eval_time'] = time.time() - start_train


