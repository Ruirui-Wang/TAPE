import os
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import time
import wandb 
import numpy as np
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN
from tqdm import tqdm 
from torch_geometric.data import Data

from graph_embed.tune_utils import mvari_str2csv, save_parmet_tune
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
from typing import Dict, Tuple




class Trainer():
    def __init__(self, 
                 FILE_PATH: str, 
                 cfg: CN, 
                 model: torch.nn.Module, 
                 emb: torch.nn.Module,
                 data: Data,
                 optimizer: torch.optim.Optimizer, 
                 splits: Dict[str, Data], 
                 run: int, 
                 repeat: int,
                 loggers: Logger, 
                 print_logger: None, 
                 device: int):
        
        self.device = device
        self.model = model.to(self.device)
        self.emb = emb
        
        # params
        self.model_name = cfg.model.type
        self.data_name = cfg.data.name
        self.FILE_PATH = FILE_PATH 
        self.name_tag = cfg.wandb.name_tag
        self.epochs = cfg.train.epochs
        self.batch_size = cfg.train.batch_size
        
        self.train_data = splits['train']
        self.test_data = splits['test']
        
        self.valid_data = splits['valid']
        self.optimizer = optimizer
        self.loggers = loggers
        self.print_logger = print_logger
        self.data_name = cfg.data.name
        # TODO change before the submission
        report_step = {
                'cora': 1,
                'pubmed': 1,
                'arxiv_2023': 1,
                'ogbn-arxiv': 1,
                'ogbn-products': 1,
        }

        self.report_step = report_step[cfg.data.name]
        
        self.model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage']
        
        self.train_func = {model_type: self._train_gae if model_type in ['GAE', 'GAT', 'GraphSage'] else self._train_vgae for model_type in self.model_types}
        self.test_func = {model_type: self._test for model_type in self.model_types}
        self.evaluate_func = {model_type: self._evaluate if model_type in ['GAE', 'GAT', 'GraphSage'] else self._evaluate_vgae for model_type in self.model_types}
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        self.run = run
        self.repeat = repeat
        self.results_rank = {}
        self.run_result = {}


    def _train_gae(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encoder(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def _train_vgae(self):
        self.model.train()
        self.optimizer.zero_grad()
        # encoder is VAE, forward is embedding
        z = self.model(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    @torch.no_grad()
    def _test(self, data: Data):
        """test"""
        self.model.eval()

        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model.encoder(self.test_data.x, self.test_data.edge_index)
        pos_y = z.new_ones(pos_edge_index.size(1)) 
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(y, pred, pos_label=1)
        return roc_auc_score(y, pred), average_precision_score(y, pred), auc(fpr, tpr)


    @torch.no_grad()
    def _evaluate(self, data: Data):
       
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model.encoder(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        
        # add a visualization of the threshold

        acc = self._acc(pos_pred, neg_pred)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})
    
        return result_mrr
    
    def _acc(self, pos_pred, neg_pred):
        hard_thres = (max(torch.max(pos_pred).item(), torch.max(neg_pred).item()) + min(torch.min(pos_pred).item(), torch.min(neg_pred).item())) / 2

        # Initialize predictions with zeros and set ones where condition is met
        y_pred = torch.zeros_like(pos_pred)
        y_pred[pos_pred >= hard_thres] = 1

        # Do the same for negative predictions
        neg_y_pred = torch.ones_like(neg_pred)
        neg_y_pred[neg_pred <= hard_thres] = 0

        # Concatenate the positive and negative predictions
        y_pred = torch.cat([y_pred, neg_y_pred], dim=0)

        # Initialize ground truth labels
        pos_y = torch.ones_like(pos_pred)
        neg_y = torch.zeros_like(neg_pred)
        y = torch.cat([pos_y, neg_y], dim=0)
        y_logits = torch.cat([pos_pred, neg_pred], dim=0)
        # Calculate accuracy    
        return (y == y_pred).float().mean().item()


    @torch.no_grad()
    def _evaluate_vgae(self, data: Data):
       
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})
    
        return result_mrr
    

    def merge_result_rank(self):
        result_test = self.evaluate_func[self.model_name](self.test_data)
        result_valid = self.evaluate_func[self.model_name](self.valid_data)
        result_train = self.evaluate_func[self.model_name](self.train_data)

        return {
            key: (result_train[key], result_valid[key], result_test[key])
            for key in result_test.keys()
        }
    
    
    def train(self):  
        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()
            
            wandb.log({'loss': loss, 'epoch': epoch}) if self.if_wandb else None
                
            if epoch % int(self.report_step) == 0:

                self.results_rank = self.merge_result_rank()
                
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {self.results_rank["AUC"][0]:.4f}, AP: {self.results_rank["AP"][0]:.4f}, MRR: {self.results_rank["MRR"][0]:.4f}, Hit@10 {self.results_rank["Hits@10"][0]:.4f}')
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_valid: {loss:.4f}, AUC: {self.results_rank["AUC"][1]:.4f}, AP: {self.results_rank["AP"][1]:.4f}, MRR: {self.results_rank["MRR"][1]:.4f}, Hit@10 {self.results_rank["Hits@10"][1]:.4f}')               
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_test: {loss:.4f}, AUC: {self.results_rank["AUC"][2]:.4f}, AP: {self.results_rank["AP"][2]:.4f}, MRR: {self.results_rank["MRR"][2]:.4f}, Hit@10 {self.results_rank["Hits@10"][2]:.4f}')               
                    
                for key, result in self.results_rank.items():
                    self.loggers[key].add_result(self.run, result)
                    
                    train_hits, valid_hits, test_hits = result
                    self.print_logger.info(
                        f'Run: {self.run + 1:02d}, Key: {key}, '
                        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')
                    
                    if self.if_wandb:
                        wandb.log({f"Metrics/train_{key}": train_hits})
                        wandb.log({f"Metrics/valid_{key}": valid_hits})
                        wandb.log({f"Metrics/test_{key}": test_hits})
                    
                self.print_logger.info('---')

        for _ in range(1):
            start_train = time.time() 
            self.train_func[self.model_name]()
            self.run_result['train_time'] = time.time() - start_train
            self.evaluate_func[self.model_name](self.test_data)
            self.run_result['eval_time'] = time.time() - start_train


    def result_statistic(self):
        result_all_run = {}
        for key in self.loggers:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list = self.loggers[key].calc_all_stats()
            
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean

            elif key == 'Hits@100':
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean

            result_all_run[key] = [mean_list, var_list]

        print(f'{best_metric_valid_str} {best_auc_valid_str}')
        print(best_metric_valid_str)
        best_auc_metric = best_valid_mean_metric
        return best_valid_mean_metric, best_auc_metric, result_all_run
    

    def save_result(self, results_dict: Dict[str, float]):  # sourcery skip: avoid-builtin-shadow
        
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, f'{self.data_name}_lm_mrr.csv')
        self.print_logger.info(f"save to {acc_file}")
        os.makedirs(root, exist_ok=True)
        mvari_str2csv(self.name_tag, results_dict, acc_file)


    def save_tune(self, results_dict: Dict[str, float], to_file: str):  # sourcery skip: avoid-builtin-shadow
        
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, to_file)
        self.print_logger.info(f"save to {acc_file}")
        os.makedirs(root, exist_ok=True)
        save_parmet_tune(self.name_tag, results_dict, acc_file)

        

