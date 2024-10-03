import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from graph_embed.tune_utils import mvari_str2csv
from ogb.linkproppred import Evaluator
from sklearn.neural_network import MLPClassifier
import argparse
import wandb
from torch_geometric import seed_everything
from pdb import set_trace as st 
import time 
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.utils.comp_budget import params_count
from graphgps.utility.utils import get_git_repo_root_path
from graphgps.score.custom_score import LinkPredictor
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, run_loop_settings,
    create_logger, set_printing
)
from graphgps.utility.utils import save_run_results_to_csv
from create_dataset import create_tfidf
from heuristic.eval import get_metric_score
from data_utils.load import load_data_lp
from graphgps.utility.utils import random_sampling, preprocess, get_average_embedding
from graphgps.train.embedding_LLM_train import Trainer_embedding_LLM
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
import re
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from create_dataset import process_texts, process_nodefeat, save_dataset, load_or_generate_datasets
from graphgps.utility.utils import config_device

"this script is used to train a link prediction based on dot, euclidean, concatenate distance followed by a MLP classifier"
FILE_PATH = f'{get_git_repo_root_path()}/'

yaml_file = {   
             'tfidf': 'core/yamls/cora/lms/tfidf.yaml',
             'w2v': 'core/yamls/cora/lms/tfidf.yaml',
             'original': 'core/yamls/cora/lms/tfidf.yaml'
            }

split_index_data = {
    'pwc_small': 1,
    'cora': 1,
    'pubmed': 1,
    'arxiv_2023': 1,
    'pwc_medium': 0.2,
    'citationv8': 0.1,
    'ogbn-arxiv': 0.2,
    'pwc_large': 0.2
}

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='ogbn-arxiv',
                        help='data name')
    parser.add_argument('--device', dest='device', required=False, 
                        help='device id', default='cpu')
    parser.add_argument('--epoch', dest='epoch', type=int, required=False,
                        default=1,
                        help='data name')
    parser.add_argument('--report_step', dest='report_step', type=int, required=False,
                        default=1,
                        help='data name')
    parser.add_argument('--embedder', dest='embedder', type=str, required=False,
                        default='original',
                        help='word embedding method')
    parser.add_argument('--decoder', dest='decoder', type=str, required=False,
                        default='dot', choices=['concat', 'dot', 'euclidean'],
                        help='word embedding method')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def get_metrics(clf, dataset, labels, evaluator_hit, evaluator_mrr):
    # Predict and calculate accuracy
    pred = clf.predict(dataset)
    # labels = np.asarray(labels)
    # acc = np.mean( labels== pred)
    
    # Calculate positive and negative predictions
    y_pos_pred = torch.tensor(pred[labels == 1])
    y_neg_pred = torch.tensor(pred[labels == 0])
    
    # Get additional metrics
    metrics = get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred)
    # metrics.update({'ACC': round(acc, 4)})
    
    return metrics


def project_main(): 
    # process params
    args = parse_args()
    args.cfg_file = yaml_file[args.embedder]
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data

    cfg.data.device = args.device
    cfg.decoder.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    cfg.embedder.type = args.embedder
    cfg.decoder.type = args.decoder
    cfg.data.scale = split_index_data[args.data]
    cfg.train.report_step = args.report_step
    loggers = create_logger(args.repeat)
    cfg.data.method = cfg.embedder.type
    print_logger = set_printing(cfg)
    
    if args.embedder == 'original':
        cfg.data.method = 'tfidf'
        
    splits, text, data = load_data_lp[cfg.data.name](cfg.data, True)
    print(f"{cfg.data.name}: num of nodes {data.num_nodes}")
    splits = random_sampling(splits, cfg.data.scale)
    
    tokenized_texts = [" ".join(preprocess(t)) for t in text]
    
    if cfg.embedder.type == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=128)
        node_features = vectorizer.fit_transform(tokenized_texts)
    elif cfg.embedder.type == 'w2v':
        model = Word2Vec(sentences=tokenized_texts, vector_size=128, window=5, min_count=1, workers=10)
        node_features = np.array([get_average_embedding(text, model) for text in tokenized_texts])
    elif cfg.embedder.type == 'original':
        node_features = data.x
    
    try:
        node_features = torch.tensor(node_features.toarray())
    except:
        node_features = torch.tensor(node_features)
        
    print_logger.info(node_features.shape)
    cfg = config_device(cfg)
    
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        set_printing(cfg)
        print(f'run id : {run_id}, seed: {seed}, split_index: {split_index}')
        
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        if cfg.decoder.type == 'concat':
            model = LinkPredictor(node_features.shape[1]*2, 2**9, 1, 3, 0.1, cfg.decoder.type).to(cfg.device)
        else: 
            model = LinkPredictor(node_features.shape[1], 2**9, 1, 3, 0.1, cfg.decoder.type).to(cfg.device)
            
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.base_lr, weight_decay=cfg.optimizer.weight_decay)
        
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

    cfg.out_dir = 'results/tfidf'
    root = os.path.join(FILE_PATH, cfg.out_dir)
    acc_file = os.path.join(root, f'{cfg.data.name}_lm_mrr.csv')
    
    os.makedirs(root, exist_ok=True)
    name_tag = cfg.wandb.name_tag = f'seperated_{cfg.data.name}_run{run_id}_{args.embedder}{args.decoder}{args.epoch}_{cfg.data.scale}'
    mvari_str2csv(name_tag, result_dict, acc_file)
    
    
if __name__ == '__main__':
    project_main()
