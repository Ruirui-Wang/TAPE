import os, sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from typing import Dict

from torch_sparse import SparseTensor
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
from graphgps.utility.utils import random_sampling

from torch_geometric import seed_everything
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
    create_optimizer, config_device, \
    create_logger

from data_utils.load import load_data_lp, load_graph_lp
from graphgps.utility.utils import save_run_results_to_csv
from yacs.config import CfgNode as CN

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
from model import BertClassifier, BertClaInfModel, NCNClaInfModel, NCNClassifier, GCNClassifier, GCNClaInfModel
from finetune_dataset import LinkPredictionDataset
from utils import init_path, time_logger
from ogb.linkproppred import Evaluator
import numpy as np
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
from torch.utils.tensorboard import SummaryWriter
from graph_embed.tune_utils import mvari_str2csv, save_parmet_tune
from graphgps.score.custom_score import mlp_score, InnerProduct
from graphgps.network.heart_gnn import GAT_Variant, GAE_forall, GCN_Variant, \
    SAGE_Variant, GIN_Variant, DGCNN

writer = SummaryWriter()


# todo
def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


def gcn_dataset(data, splits):
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    # Use training + validation edges for inference on test set.
    val_edge_index = splits['valid']['pos_edge_label_index']
    full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
    data.full_adj_t = SparseTensor.from_edge_index(full_edge_index,
                                                   sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
    data.full_adj_t = data.full_adj_t.to_symmetric()
    return data


def create_GAE_model(cfg_model: CN,
                     cfg_score: CN,
                     model_name: str):
    if model_name in {'GAT', 'VGAE', 'GAE', 'GraphSage'}:
        raise NotImplementedError('Current model does not exist')
        # model = create_model(cfg_model)

    elif model_name == 'GAT_Variant':
        encoder = GAT_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              cfg_model.heads,
                              )
    elif model_name == 'GCN_Variant':
        encoder = GCN_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              )
    elif model_name == 'SAGE_Variant':
        encoder = SAGE_Variant(cfg_model.in_channels,
                               cfg_model.hidden_channels,
                               cfg_model.out_channels,
                               cfg_model.num_layers,
                               cfg_model.dropout,
                               )
    elif model_name == 'GIN_Variant':
        encoder = GIN_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              cfg_model.mlp_layer
                              )

    if cfg_score.product == 'dot':
        decoder = mlp_score(cfg_model.out_channels,
                            cfg_score.score_hidden_channels,
                            cfg_score.score_out_channels,
                            cfg_score.score_num_layers,
                            cfg_score.score_dropout,
                            cfg_score.product)
    elif cfg_score.product == 'inner':
        decoder = InnerProduct()

    else:
        # Without this else I got: UnboundLocalError: local variable 'model' referenced before assignment
        raise ValueError('Current model does not exist')

    return GAE_forall(encoder=encoder, decoder=decoder)


class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink
        self.decoder = cfg.decoder

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
        self.data = gcn_dataset(self.data, splits)

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
        if 'mpnet' not in cfg.lm.model.name:
            bert_model.gradient_checkpointing_enable()
        hidden_size = bert_model.config.hidden_size
        current_size = self.data.x.size(1)

        if current_size < hidden_size:
            padding_size = hidden_size - current_size
            self.data.x = F.pad(self.data.x, (0, padding_size), "constant", 0)
        elif current_size > hidden_size:
            self.data.x = self.data.x[:, :hidden_size]

        for name, param in bert_model.named_parameters():
            if 'encoder.layer.11' in name and 'mpnet' in cfg.lm.model.name:
                break
            if 'encoder.layer.5' in name and 'MiniLM' in cfg.lm.model.name:
                break
            if 'layers.31' in name and 'Llama' in cfg.lm.model.name:
                break
            if 'encoder.layer.11' in name and 'bert' in cfg.lm.model.name:
                break
            if 'encoder.layer.23' in name and 'e5-large' in cfg.lm.model.name:
                break
            param.requires_grad = False

        if self.decoder.model.type == 'MLP':
            self.model = BertClassifier(bert_model, cfg, feat_shrink=self.feat_shrink).to(self.device)
        elif self.decoder.model.type == 'NCN' or self.decoder.model.type == 'NCNC':
            self.model = NCNClassifier(bert_model, cfg, self.data, self.data.edge_index).to(self.device)
        elif self.decoder.model.type == 'GCN_Variant':
            cfg_model = eval(f'cfg.decoder.model.{args.model}')
            cfg_score = eval(f'cfg.decoder.score.{args.model}')
            cfg_model.in_channels = hidden_size
            self.model = GCNClassifier(bert_model, cfg, self.data, self.data.edge_index,
                                       create_GAE_model(cfg_model, cfg_score, args.model)).to(self.device)
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        self.tensorboard_writer = writer
        self.loggers = create_logger(args.repeat)
        self.print_logger = set_printing(cfg)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(f'{n} is trainable.')
        self.trainable_params = sum(p.numel()
                                    for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable params: {self.trainable_params}')
        self.name_tag = cfg.model.type + '-' + cfg.data.name + '-' + cfg.decoder.model.type
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
            remove_unused_columns=False if self.decoder.model.type != 'MLP' else True
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

        if self.decoder.model.type == 'MLP':
            inf_model = BertClaInfModel(
                self.model, emb, pred, feat_shrink=self.feat_shrink)
        elif self.decoder.model.type == 'NCN' or self.decoder.model.type == 'NCNC':
            inf_model = NCNClaInfModel(
                self.model, emb, pred, self.data, self.data.edge_index, feat_shrink=self.feat_shrink)
        elif self.decoder.model.type == 'GCN_Variant':
            inf_model = GCNClaInfModel(
                self.model, emb, pred, self.data, self.data.edge_index, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size * 1,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=False,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        predictor_dict = trainer.predict(eval_data)
        pred = predictor_dict.predictions
        pred = pred.squeeze()
        pos_mask = (predictor_dict.label_ids == 1)
        neg_mask = (predictor_dict.label_ids == 0)

        pos_pred = predictor_dict.predictions[pos_mask]
        neg_pred = predictor_dict.predictions[neg_mask]
        pos_pred = torch.tensor(pos_pred, dtype=torch.float32)
        neg_pred = torch.tensor(neg_pred, dtype=torch.float32)
        pos_pred = pos_pred.squeeze()
        neg_pred = neg_pred.squeeze()

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
    parser.add_argument('--decoder', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    if args.decoder:
        cfg_decoder = set_cfg(FILE_PATH, args.decoder)
        cfg.decoder = cfg_decoder
    else:
        from yacs.config import CfgNode as CN

        cfg.decoder = CN()
        cfg.decoder.model = CN()
        cfg.decoder.model.type = 'MLP'
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
        trainer = LMTrainer(cfg)
        start_inf = time.time()
        trainer.train()
        eval_time = time.time() - start_inf
        result_test = trainer.eval_and_save(trainer.test_dataset)
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

    print('All runs:')

    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = loggers[key].calc_all_stats()
        result_dict[key] = valid_test

    trainer.save_result(result_dict)

    print_logger.info(f"Results for: {cfg.model.type}")
    print_logger.info(f"Model Params: {trainer.trainable_params}")
    print_logger.info(f"Training time: {eval_time}")


