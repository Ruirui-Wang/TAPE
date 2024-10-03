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
from torch_geometric import seed_everything
from pdb import set_trace as st 
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, run_loop_settings,
    create_logger
)
# from cuml.ensemble import RandomForestClassifier as cuRF
from heuristic.eval import get_metric_score
import nltk
from graphgps.score.custom_score import LinkPredictor, mlp_decoder
from nltk.tokenize import word_tokenize
import re
from gensim.models import Word2Vec
from graphgps.utility.utils import (
    set_cfg, parse_args, set_printing, 
    random_sampling, preprocess, get_average_embedding
)
import time 
from torch_geometric.graphgym.utils.comp_budget import params_count
from data_utils.load import load_data_lp
from create_dataset import process_texts, process_nodefeat, load_or_generate_datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from graphgps.train.embedding_LLM_train import Trainer_Triples
from graphgps.utility.utils import save_run_results_to_csv
"this script aims to compare rf and mlp based on concatenated embeddings"
FILE_PATH = f'{get_git_repo_root_path()}/'

yaml_file = {   
             'tfidf': 'core/yamls/cora/lms/tfidf.yaml',
             'w2v': 'core/yamls/cora/lms/tfidf.yaml',
            }

split_index_data = {
    'pwc_small': 1,
    'cora': 1,
    'pubmed': 1,
    'arxiv_2023': 1,
    'pwc_medium': 0.5,
    'citationv8': 0.5,
    'ogbn_arxiv': 0.5,
    'pwc_large': 0.5
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
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='arxiv_2023',
                        help='data name')
    parser.add_argument('--device', dest='device', required=False, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=200,
                        help='data name')
    parser.add_argument('--embedder', dest='embedder', type=str, required=False,
                        default='w2v',
                        help='word embedding method')
    # parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
    #                     help='score')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

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

    writer = SummaryWriter()
    cfg.run_dir = writer.log_dir
    
    cfg.data.scale = split_index_data[args.data]
    
    loggers = create_logger(args.repeat)
    cfg.data.method = cfg.embedder.type
    print_logger = set_printing(cfg)
    
    splits, text, _ = load_data_lp[cfg.data.name](cfg.data)
    splits = random_sampling(splits, cfg.data.scale)

    if cfg.data.name in ['pwc_small', 'pwc_medium', 'pwc_large', 'citationv8']:
        text = text['feat'].tolist()
    
    train_data, train_labels = process_texts(splits['train'].pos_edge_label_index,
                                             splits['train'].neg_edge_label_index, 
                                             text)
    val_data, val_labels = process_texts(splits['valid'].pos_edge_label_index,
                                             splits['valid'].neg_edge_label_index, 
                                             text)
    test_data, test_labels = process_texts(splits['test'].pos_edge_label_index,
                                             splits['test'].neg_edge_label_index, 
                                             text)
    
    start_time = time.time()
    if cfg.embedder.type == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=128)
        train_triples = vectorizer.fit_transform(train_data)
        val_triples = vectorizer.fit_transform(val_data)
        test_triples = vectorizer.fit_transform(test_data)
    elif cfg.embedder.type == 'w2v':
        model = Word2Vec(sentences=train_data+val_data+test_data, vector_size=128, window=5, min_count=1, workers=10)
        train_triples = torch.tensor([get_average_embedding(text, model) for text in train_data])
        val_triples = torch.tensor([get_average_embedding(text, model) for text in val_data])
        test_triples = torch.tensor([get_average_embedding(text, model) for text in test_data])
    elif cfg.embedder.type == 'original':
        exit(-1)
        
    data = {'train': (train_triples, train_labels),
            'valid': (val_triples, val_labels),
            'test': (test_triples, test_labels)}
    
    print(f"Embedding time: {time.time() - start_time}")
    
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        
        print(f'run id : {run_id}, seed: {seed}, split_index: {split_index}')
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        model = mlp_decoder(128, 2**9, 1, 3, 0.1)
            
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.base_lr, weight_decay=cfg.optimizer.weight_decay)
        trainer = Trainer_Triples(FILE_PATH,
                                cfg,
                                model,
                                optimizer,
                                data,
                                run_id,
                                args.repeat,
                                loggers,
                                print_logger=print_logger,
                                batch_size=cfg.train.batch_size)


        start = time.time()
        trainer.train()
        end = time.time()
        print('Training time: ', end - start)
        save_run_results_to_csv(cfg, trainer.loggers, seed, run_id)


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
    name_tag = cfg.wandb.name_tag = f'combined_{cfg.data.name}_run{run_id}_{args.embedder}_nodecoder_{args.epoch}_{cfg.data.scale}'
    mvari_str2csv(name_tag, result_dict, acc_file)
    
    
if __name__ == '__main__':
    project_main()
