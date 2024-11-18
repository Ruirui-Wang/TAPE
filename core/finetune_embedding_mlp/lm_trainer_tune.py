import os, sys
from typing import Dict

from tqdm import tqdm
import itertools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
from graphgps.utility.utils import random_sampling

from torch_geometric import seed_everything
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, config_device, \
    create_logger

from data_utils.load import load_data_lp
from graphgps.utility.utils import save_run_results_to_csv

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from model import BertClassifier, BertClaInfModel
from finetune_dataset import LinkPredictionDataset
from utils import init_path, time_logger
from ogb.linkproppred import Evaluator
import numpy as np

from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device
from torch.utils.tensorboard import SummaryWriter
from graph_embed.tune_utils import mvari_str2csv

writer = SummaryWriter()

# todo
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
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr
        self.device = config_device(cfg).device

        self.use_gpt_str = "2" if cfg.lm.train.use_gpt else ""
        self.output_dir = f'output/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'

        # Preprocess data
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)
        splits = random_sampling(splits, args.downsampling)

        self.data = data.to(self.device)
        self.num_nodes = data.num_nodes
        self.n_labels = 2

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if 'llama' in cfg.lm.model.name:
            tokenizer.pad_token = tokenizer.eos_token
            X = tokenizer(text, padding=True, truncation=True, max_length=512)
        else:
            X = tokenizer(text, padding=True, truncation=True, max_length=512)
        dataset = LinkPredictionDataset(X, data.edge_index, torch.ones(data.edge_index.shape[1]))
        self.inf_dataset = dataset

        self.train_dataset = LinkPredictionDataset(X, torch.cat(
            [splits['train'].pos_edge_label_index, splits['train'].neg_edge_label_index], dim=1), torch.cat(
            [splits['train'].pos_edge_label, splits['train'].neg_edge_label], dim=0))
        self.val_dataset = LinkPredictionDataset(X, torch.cat(
            [splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index], dim=1), torch.cat(
            [splits['valid'].pos_edge_label, splits['valid'].neg_edge_label], dim=0))
        self.test_dataset = LinkPredictionDataset(X, torch.cat(
            [splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index], dim=1), torch.cat(
            [splits['test'].pos_edge_label, splits['test'].neg_edge_label], dim=0))

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name, attn_implementation="eager")
        for name, param in bert_model.named_parameters():
            if 'encoder.layer.5' in name and 'minilm' in cfg.lm.model.name:
                break
            if 'layers.31' in name and 'Llama' in cfg.lm.model.name:
                break
            param.requires_grad = False
        self.model = BertClassifier(bert_model,
                                    cfg,
                                    feat_shrink=self.feat_shrink).to(self.device)

        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        self.tensorboard_writer = writer
        self.loggers = create_logger(args.repeat)
        self.print_logger = set_printing(cfg)

        self.trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        self.name_tag = cfg.model.type + cfg.data.name
        self.FILE_PATH = f'{get_git_repo_root_path()}/'

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 1
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
            per_device_eval_batch_size=self.batch_size * 8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True,
            max_grad_norm=10.0,
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
    def eval_and_save(self, eval_data):
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
            per_device_eval_batch_size=self.batch_size * 8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=False,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        predictor_dict = trainer.predict(eval_data)
        pos_mask = (predictor_dict.label_ids == 1)
        neg_mask = (predictor_dict.label_ids == 0)

        pos_pred = predictor_dict.predictions[pos_mask]
        neg_pred = predictor_dict.predictions[neg_mask]
        pos_pred = torch.tensor(pos_pred, dtype=torch.float32)
        neg_pred = torch.tensor(neg_pred, dtype=torch.float32)

        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': 0.00})
        return result_mrr

    def save_result(self, results_dict: Dict[str, float]):  # sourcery skip: avoid-builtin-shadow

        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, f'{self.dataset_name}_lm_mrr.csv')
        self.print_logger.info(f"save to {acc_file}")
        os.makedirs(root, exist_ok=True)
        mvari_str2csv(self.name_tag, results_dict, acc_file)


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/lms/ft-deberta.yaml',
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
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
    cfg.model.device = args.device
    cfg.device = args.device
    torch.set_num_threads(cfg.num_threads)
    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
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

        hyperparameter_search = {"lr": [0.001, 0.0005, 0.0001, 0.00005], 'grad_acc_steps': [1, 2],'warmup_epochs': [0.6, 0.8, 1.0],
                                 'dropout': [0.0, 0.1, 0.2, 0.3], 'weight_decay': [0.0, 0.1, 0.2], 'att_dropout': [0.0, 0.1, 0.2]}

        print_logger.info(f"hypersearch space: {hyperparameter_search}")
        for lr, grad_acc_steps, warmup_epochs, dropout, weight_decay, hidden_channels, num_layers, att_dropout in tqdm(
                itertools.product(*hyperparameter_search.values())):
            cfg.lm.train.lr = lr
            cfg.lm.train.grad_acc_steps = grad_acc_steps
            cfg.lm.train.warmup_epochs = warmup_epochs
            cfg.lm.train.dropout = dropout
            cfg.lm.train.weight_decay = weight_decay
            cfg.lm.train.att_dropout = att_dropout
            print_logger.info(f"lr: {lr}, grad_acc_steps: {grad_acc_steps}, warmup_epochs: {warmup_epochs}, "
                              f"dropout: {dropout}, weight_decay: {weight_decay}, hidden_channels: {hidden_channels}, "
                              f"num_layers: {num_layers}, att_dropout: {att_dropout}")
            trainer = LMTrainer(cfg)
            trainer.train()
            start_inf = time.time()
            result_test = trainer.eval_and_save(trainer.test_dataset)
            eval_time = time.time() - start_inf
            result_valid = trainer.eval_and_save(trainer.val_dataset)
            result_train = trainer.eval_and_save(trainer.train_dataset)
            result_all = {
                key: (result_train[key], result_valid[key], result_test[key])
                for key in result_test.keys()
            }
            for key, result in result_all.items():
                loggers[key].add_result(run_id, result)

                trainer.tensorboard_writer.add_scalar(f"Metrics/Train/{key}", result[0])
                trainer.tensorboard_writer.add_scalar(f"Metrics/Valid/{key}", result[1])
                trainer.tensorboard_writer.add_scalar(f"Metrics/Test/{key}", result[2])

                train_hits, valid_hits, test_hits = result
                trainer.print_logger.info(
                    f'Run: {run_id + 1:02d}, Key: {key}, '
                    f'Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')

            trainer.print_logger.info('---')
            save_run_results_to_csv(cfg, loggers, seed, run_id)

            run_result = {}
            for key in trainer.loggers.keys():
                if trainer.loggers[key].results == [[], []]:
                    run_result[key] = None
                else:
                    # refer to calc_run_stats in Logger class
                    _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                    run_result[key] = test_bvalid

            run_result.update(
                {'lr': lr, 'grad_acc_steps': grad_acc_steps, 'warmup_epochs': warmup_epochs, 'dropout': dropout,
                    'weight_decay': weight_decay, 'hidden_channels': hidden_channels, 'num_layers': num_layers,
                    'att_dropout': att_dropout}
            )
            print_logger.info(run_result)

            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)
            torch.cuda.empty_cache()





