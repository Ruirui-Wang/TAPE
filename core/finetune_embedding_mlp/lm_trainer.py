import copy
import os, sys

import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.graphgym import params_count


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
from functools import partial
from graphgps.utility.utils import random_sampling

from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
    create_optimizer, config_device, \
    create_logger

from graphgps.utility.ncn import PermIterator
from graphgps.network.ncn import predictor_dict, convdict, GCN
from data_utils.load import load_data_lp, load_graph_lp
from graphgps.train.ncn_train import Trainer_NCN
from graphgps.utility.utils import save_run_results_to_csv


from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from model import BertClassifier, BertClaInfModel
from finetune_dataset import LinkPredictionDataset
from utils import init_path, time_logger

def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.use_gpt_str = "2" if cfg.lm.train.use_gpt else ""
        self.output_dir = f'output/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'

        # Preprocess data
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)
        splits = random_sampling(splits, args.downsampling)

        self.data = data
        self.num_nodes = data.num_nodes
        self.n_labels = 2

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        dataset = LinkPredictionDataset(X, data.edge_index, np.ones(data.edge_index.shape[1]))
        self.inf_dataset = dataset

        self.train_dataset = LinkPredictionDataset(X, torch.cat([splits['train'].pos_edge_label_index, splits['train'].neg_edge_label_index], dim=1), torch.cat([splits['train'].pos_edge_label, splits['train'].neg_edge_label], dim=0))
        self.val_dataset = LinkPredictionDataset(X, torch.cat([splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index], dim=1), torch.cat([splits['valid'].pos_edge_label, splits['valid'].neg_edge_label], dim=0))
        self.test_dataset = LinkPredictionDataset(X, torch.cat([splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index], dim=1), torch.cat([splits['test'].pos_edge_label, splits['test'].neg_edge_label], dim=0))


        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=False,
            dataloader_drop_last=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train pre-trained model
        self.trainer.train()
        torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)
        else:
            from core.GNNs.gnn_utils import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels): return _evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        def eval(x): return evaluator(
            np.argmax(pred[x], -1), self.data.y[x])

        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/lms/ft-deberta.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', type=str, required=False, default='cora',
                        help='data name')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    parser.add_argument('--device', dest='device', required=False,
                        help='device id')
    parser.add_argument('--downsampling', type=float, default=1,
                        help='Downsampling rate.')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=1000)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()



if __name__ == '__main__':
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.device = args.device
    cfg.device = args.device
    torch.set_num_threads(cfg.num_threads)
    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    cfg.device = args.device
    start_ft = time.time()
    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        cfg.seed = seed
        trainer = LMTrainer(cfg)
        trainer.train()


