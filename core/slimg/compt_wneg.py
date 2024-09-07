import os
import sys

# Add parent directory to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
This script implements the compatibility matrix method with and without negative links,
as described in the paper "NETINFOF FRAMEWORK: MEASURING AND EXPLOITING NETWORK USABLE INFORMATION".
"""
import os
import sys
import torch
import numpy as np
import scipy.sparse as ssp
import matplotlib.pyplot as plt
from pathlib import Path
from IPython import embed
import wandb
from scipy.spatial import distance
from sklearn.neural_network import MLPClassifier
from yacs.config import CfgNode as CN
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from core.graphgps.utility.utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)
from data_utils.load_cora_lp import get_cora_casestudy
from data_utils.load_arxiv2023_lp import get_raw_text_arxiv_2023
from data_utils.load_pubmed_lp import get_pubmed_casestudy
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj
from heuristic.eval import get_metric_score
from graph_embed.tune_utils import parse_args, param_tune_acc_mrr
import torch
from time import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm 
from core.slimg.mlp_dot_product import distance_metric, data_loader, FILE_PATH, set_cfg
import torch.nn.functional as F
import torch

method = 'compat'
LOG_FREQ = 10 

norm = lambda x: (x-x.mean())/ x.std()

def create_node_feat(edge_label_index, train_node_feat):
    node_feat = torch.zeros(edge_label_index.shape[0], train_node_feat.shape[1]*2)
    for i, (srt, trg) in enumerate(edge_label_index):
        node_feat[i, :] = torch.cat((train_node_feat[srt], train_node_feat[trg]), 0)
    
    return node_feat


class MultipleLinearRegression_with_NegLink(torch.nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim):
        super(MultipleLinearRegression_with_NegLink, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x1, x2):
        xH = self.linear1(x1)
        xHx = xH@x2.T
        return xHx
    
    def predictor(self, x1):
        x = self.linear1(x1)
        return x
        
    

class MSELoss_w_Neg(torch.nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss_w_Neg, self).__init__()
        
        
    def forward(self, predicted, target):
        # Compute your custom loss here
        pos = target == 1
        pos_loss = (1 - predicted[pos, :]).sum()
        
        neg = target == 0
        neg_loss = (predicted[neg, :]).sum()
        
        return neg_loss + pos_loss


class Trainer():

    def __init__(self, 
                 cfg, 
                 ):
        # self.seed = cfg.seed
        # self.device = cfg.device
        self.dataset_name = cfg.data.name
        self.model_name = 'comp'
        
        self.lr = 0.015
        self.epochs = 200
        self.batch_size = 256

        if torch.cuda.is_available():
            # Get the number of available CUDA devices
            num_cuda_devices = torch.cuda.device_count()
        else:
            num_cuda_devices = 0
        if num_cuda_devices > 0:
            # Set the first CUDA device as the active device
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            self.device = 'cpu'
                
        self.id = wandb.util.generate_id()
        # preprocess data
        dataset, data_cited, splits = data_loader[cfg.data.name](cfg)
        
        # use test edge_index as full_A
        full_edge_index = splits['test'].edge_index
        full_edge_weight = torch.ones(full_edge_index.size(1))
        num_nodes = dataset._data.num_nodes
        
        m = construct_sparse_adj(full_edge_index)
        plot_coo_matrix(m, f'test_edge_index.png')


        self.features = splits['train'].x
        
        self.train_links = splits['train'].edge_label_index
        self.train_labels = splits['train'].edge_label
        self.valid_links = splits['valid'].edge_label_index 
        self.valid_labels = splits['valid'].edge_label
        self.test_links = splits['test'].edge_label_index
        self.test_labels = splits['test'].edge_label
        # self.
        self.test_loader = DataLoader(range(self.test_links.shape[1]), self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(range(self.valid_links.shape[1]), self.batch_size, shuffle=False)
        self.train_loader = DataLoader(range(self.train_links.shape[1]), self.batch_size, shuffle=False)
        
        num_feat = self.features.shape[1]
        # models
        self.model = MultipleLinearRegression_with_NegLink(num_feat, num_feat).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        
        if cfg.train.loss == 'mse':
            self.loss_func = torch.nn.MSELoss()
        elif cfg.train.loss == 'mse_w_neg':
            self.loss_func = MSELoss_w_Neg()
            
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')


    def _train(self):
        # ! Shared
        self.model.train()
        for batch_count, indices in enumerate(tqdm(self.train_loader)):
            # do node level things
            emb = (self.features.to(self.device))
        
            self.optimizer.zero_grad()
            # ! Specific
            links_indices = self.train_links[:, indices]
            labels = self.train_labels[indices]
            zHz = self.model.forward(emb[links_indices[0]], emb[links_indices[1]])
            
            loss = self.loss_func(zHz, labels)

            loss.backward()
            self.optimizer.step()
            del emb, batch_count, indices 
        return loss.item()


    def _link_predictor(self, emb1, emb2):
        """ calc the similarity between two node embeddings 
        # previous version is
        # def _link_predictor(self, emb1, emb2):
        #   return F.cosine_similarity(norm(emb1), norm(emb2), dim=0)
        """
        # if self.device == 'cpu'
        link_pred = torch.zeros(emb1.shape[0])
        
        for index in range(emb1.shape[0]):
            link_pred[index] = distance_metric['cos'](emb1[index].cpu(), emb2[index].cpu())
        return torch.tensor(link_pred)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        emb = self.features.to(self.device)
        
        def evaluate_loader(loader, labels):
            pred_list = []
            for _, indices in enumerate(tqdm(loader)):
                links_indices = labels[:, indices]
                emb_right = self.model.predictor(emb[links_indices[0]])
                pred_list.append(self._link_predictor(emb_right, emb[links_indices[1]]))
            predictions = torch.cat(pred_list)
            return predictions
        
        train_predictions = evaluate_loader(self.train_loader, self.train_links)
        valid_predictions = evaluate_loader(self.valid_loader, self.valid_links)
        test_predictions = evaluate_loader(self.test_loader, self.test_links)

        def compute_metrics(predictions, labels):
            y_pos_pred = predictions[labels == 1]
            y_neg_pred = predictions[labels == 0]
            mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, y_pos_pred, y_neg_pred)
            mrr.update({})
            
            threshold = (predictions.max() + predictions.min()) / 2
            predictions = torch.where(predictions >= threshold, 1, 0)
            
            accuracy = torch.sum(predictions == labels).float() / labels.shape[0]
            mrr.update({'ACC': accuracy.tolist()})
            
            return accuracy, mrr
        
        train_acc, train_mrr = compute_metrics(train_predictions, self.train_labels)
        valid_acc, valid_mrr = compute_metrics(valid_predictions, self.valid_labels)
        test_acc, test_mrr = compute_metrics(test_predictions, self.test_labels)
        
        return train_acc, train_mrr, valid_acc, valid_mrr, test_acc, test_mrr
        
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss = self._train()
            
            if epoch % LOG_FREQ == 0:
                train_acc, train_mrr, valid_acc, valid_mrr, test_acc, test_mrr = self._evaluate()
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, Train_acc: {train_acc:.4f}, Train_mrr: {train_mrr}, ')
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, Valid_acc: {valid_acc:.4f}, Valid_mrr: {valid_mrr}, ')
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, Test_acc: {test_acc:.4f}, Test_mrr: {test_mrr}, ')
        return self.model

    @torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, val_mrr = self._evaluate()
        print(
            f'[{self.model_name}] ValAcc: {val_acc:.4f}, \n')
        return 


    def save_weight(self):
        plt.figure()
        weight = self.model.linear1.weight.cpu().detach().numpy()
        plt.imshow( norm(weight))
        plt.savefig(f'{cfg.data.name}_compt_with_neg_{self.id}.png')
        return 
      


def analysis_H(cfg) -> None:
    """load text attribute graph in link predicton setting
    """

    MLR_model = Trainer(cfg)
    # train loop 
    MLR_model.train()
    # evaluate loop 

    root = FILE_PATH + 'results/node_sim/'
    acc_file = root + f'{cfg.data.name}_acc_{method}.csv'

    if not os.path.exists(root):
        Path(root).mkdir(parents=True, exist_ok=False)
    
    
    MLR_model.save_weight()
    
    return 


if __name__ == "__main__":
    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)


    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    result = analysis_H(cfg)
    print(result)
