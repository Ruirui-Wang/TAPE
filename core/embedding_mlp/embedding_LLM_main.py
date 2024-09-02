import copy
import os, sys
import gc
import transformers
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch_geometric.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, \
    create_optimizer, config_device, \
    create_logger
from torch_geometric.graphgym.utils.comp_budget import params_count
from data_utils.load import load_data_lp, load_graph_lp
from graphgps.train.embedding_LLM_train import Trainer_embedding_LLM, Trainer_embedding_LLM_Cross
from graphgps.utility.utils import save_run_results_to_csv, random_sampling
from graphgps.utility.utils import random_sampling
from graphgps.score.custom_score import LinkPredictor, mlp_decoder


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/pwc_small/lms/minilm.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', type=str, required=False, default='pwc_small',
                        help='data name')
    parser.add_argument('--product', type=str, required=False, default='dot',
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
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
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
    cfg.train.epochs = args.epoch
    cfg.model.product = args.product

    torch.set_num_threads(cfg.num_threads)
    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    cfg.device = args.device
    splits, text, data = load_data_lp[cfg.data.name](cfg.data)
    splits = random_sampling(splits, args.downsampling)
    start_emb = time.time()
    if cfg.model.product == 'cross':
        cfg.wandb.name_tag = cfg.wandb.name_tag + '_cross'
        saved_features_path = './' + cfg.embedder.type + cfg.data.name + '_cross_saved_edge_features.pt'
        if os.path.exists(saved_features_path):
            edge_features = torch.load(saved_features_path)
            splits['train'].edge_features = edge_features[0]
            splits['valid'].edge_features = edge_features[1]
            splits['test'].edge_features = edge_features[2]
        else:
            train_edge_features = []
            valid_edge_features = []
            test_edge_features = []
            train_edges = torch.cat((splits['train']['pos_edge_label_index'], splits['train']['neg_edge_label_index']), dim=1)
            valid_edges = torch.cat((splits['valid']['pos_edge_label_index'], splits['valid']['neg_edge_label_index']), dim=1)
            test_edges = torch.cat((splits['test']['pos_edge_label_index'], splits['test']['neg_edge_label_index']), dim=1)
            train_edge_text = [text[edge[0]] + text[edge[1]] for edge in train_edges.T]
            valid_edge_text = [text[edge[0]] + text[edge[1]] for edge in valid_edges.T]
            test_edge_text = [text[edge[0]] + text[edge[1]] for edge in test_edges.T]
            if cfg.embedder.type == 'minilm':
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=cfg.device)
                train_edge_features = model.encode(train_edge_text, batch_size=256)
                valid_edge_features = model.encode(valid_edge_text, batch_size=256)
                test_edge_features = model.encode(test_edge_text, batch_size=256)
            elif cfg.embedder.type == 'e5-large':
                model = SentenceTransformer('intfloat/e5-large-v2', device=cfg.device)
                train_edge_features = model.encode(train_edge_text, normalize_embeddings=True, batch_size=256)
                valid_edge_features = model.encode(valid_edge_text, normalize_embeddings=True, batch_size=256)
                test_edge_features = model.encode(test_edge_text, normalize_embeddings=True, batch_size=256)
            elif cfg.embedder.type == 'llama':
                from huggingface_hub import HfFolder

                token = os.getenv("HUGGINGFACE_HUB_TOKEN")
                HfFolder.save_token(token)
                model_id = "meta-llama/Meta-Llama-3-8B"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.pad_token = tokenizer.eos_token
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                node_features = []
                batch_size = 64
                for i in range(0, len(train_edge_text), batch_size):
                    batch_texts = train_edge_text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.last_hidden_state
                        batch_features = torch.mean(batch_features, dim=1)
                        train_edge_features.append(batch_features)
                train_edge_features = torch.cat(train_edge_features, dim=0)
                train_edge_features = train_edge_features.to(torch.float32)
                for i in range(0, len(valid_edge_text), batch_size):
                    batch_texts = valid_edge_text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.last_hidden_state
                        batch_features = torch.mean(batch_features, dim=1)
                        valid_edge_features.append(batch_features)
                valid_edge_features = torch.cat(valid_edge_features, dim=0)
                valid_edge_features = valid_edge_features.to(torch.float32)
                for i in range(0, len(test_edge_text), batch_size):
                    batch_texts = test_edge_text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.last_hidden_state
                        batch_features = torch.mean(batch_features, dim=1)
                        test_edge_features.append(batch_features)
                test_edge_features = torch.cat(test_edge_features, dim=0)
                test_edge_features = test_edge_features.to(torch.float32)

            elif cfg.embedder.type == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertModel.from_pretrained("bert-base-uncased").to(cfg.device)
                node_features = []
                batch_size = 256

                for i in range(0, len(train_edge_text), batch_size):
                    batch_texts = train_edge_text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.pooler_output
                        train_edge_features.append(batch_features)
                train_edge_features = torch.cat(train_edge_features, dim=0)
                for i in range(0, len(valid_edge_text), batch_size):
                    batch_texts = valid_edge_text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.pooler_output
                        valid_edge_features.append(batch_features)
                valid_edge_features = torch.cat(valid_edge_features, dim=0)
                for i in range(0, len(test_edge_text), batch_size):
                    batch_texts = test_edge_text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.pooler_output
                        test_edge_features.append(batch_features)
                test_edge_features = torch.cat(test_edge_features, dim=0)
            edge_features = [train_edge_features, valid_edge_features, test_edge_features]
            torch.save(edge_features, saved_features_path)
        emb_time = time.time() - start_emb
        splits['train'].edge_features = edge_features[0]
        splits['valid'].edge_features = edge_features[1]
        splits['test'].edge_features = edge_features[2]

        splits['train'].edge_features = torch.tensor(splits['train'].edge_features)
        splits['valid'].edge_features = torch.tensor(splits['valid'].edge_features)
        splits['test'].edge_features = torch.tensor(splits['test'].edge_features)


        for run_id in range(args.repeat):
            seed = run_id + args.start_seed
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg)
            print_logger = set_printing(cfg)
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            cfg = config_device(cfg)

            print_logger.info("start training")
            print(cfg.data.name)
            model = mlp_decoder(edge_features[0].shape[1], cfg.model.hidden_channels, 1, cfg.model.num_layers, cfg.model.dropout)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.base_lr,
                                         weight_decay=cfg.optimizer.weight_decay)
            trainer = Trainer_embedding_LLM_Cross(FILE_PATH,
                                            cfg,
                                            model,
                                            optimizer,
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
    else:
        cfg.wandb.name_tag = cfg.wandb.name_tag + '_' + cfg.model.product
        saved_features_path = './' + cfg.embedder.type + '_' + cfg.data.name + '_saved_node_features.pt'
        if os.path.exists(saved_features_path):
            node_features = torch.load(saved_features_path)
        else:
            if cfg.embedder.type == 'minilm':
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=cfg.device)
                emb_params = params_count(model)
                node_features = model.encode(text, batch_size=256)
            elif cfg.embedder.type == 'e5-large':
                model = SentenceTransformer('intfloat/e5-large-v2', device=cfg.device)
                emb_params = params_count(model)
                node_features = model.encode(text, normalize_embeddings=True, batch_size=256)
            elif cfg.embedder.type == 'llama':
                from huggingface_hub import HfFolder

                token = os.getenv("HUGGINGFACE_HUB_TOKEN")
                HfFolder.save_token(token)
                model_id = "meta-llama/Meta-Llama-3-8B"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.pad_token = tokenizer.eos_token
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                emb_params = params_count(model)
                node_features = []
                batch_size = 64
                for i in range(0, len(text), batch_size):
                    batch_texts = text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.last_hidden_state
                        batch_features = torch.mean(batch_features, dim=1)
                        node_features.append(batch_features)
                node_features = torch.cat(node_features, dim=0)
                node_features = node_features.to(torch.float32)
            elif cfg.embedder.type == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertModel.from_pretrained("bert-base-uncased").to(cfg.device)
                emb_params = params_count(model)
                node_features = []
                batch_size = 256
                for i in range(0, len(text), batch_size):
                    batch_texts = text[i:i + batch_size]
                    encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                              max_length=512).to(cfg.device)
                    with torch.no_grad():
                        outputs = model(**encoded_input)
                        batch_features = outputs.pooler_output
                        print(batch_features.shape)
                        node_features.append(batch_features)
                node_features = torch.cat(node_features, dim=0)
            elif cfg.embedder.type == 'gemma':
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
                model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="auto")
                emb_params = params_count(model)
                node_features = []
                batch_size = 16
                for i in range(0, len(text), batch_size):
                    batch_texts = text[i:i + batch_size]
                    input_ids = tokenizer(batch_texts, return_tensors="pt", padding='max_length', truncation=True, max_length=512).to("cuda")
                    with torch.no_grad():
                        batch_features = model.get_input_embeddings()(input_ids.input_ids)
                        batch_features = torch.mean(batch_features, dim=1)
                        print(batch_features)
                        node_features.append(batch_features)
                node_features = torch.cat(node_features, dim=0)
                node_features = node_features.to(torch.float32)
            node_features = torch.tensor(node_features)
            torch.save(node_features, saved_features_path)
        emb_time = time.time() - start_emb

        node_features = torch.tensor(node_features)
        print(node_features.shape)

        for run_id in range(args.repeat):
            seed = run_id + args.start_seed
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg)
            print_logger = set_printing(cfg)
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            cfg = config_device(cfg)

            print_logger.info("start training")
            print_logger.info(node_features.shape)
            print(cfg.data.name)
            if cfg.model.product == 'concat':
                model = LinkPredictor(node_features.shape[1]+node_features.shape[1], cfg.model.hidden_channels, 1, cfg.model.num_layers,
                                      cfg.model.dropout, cfg.model.product)
            else:
                model = LinkPredictor(node_features.shape[1], cfg.model.hidden_channels, 1, cfg.model.num_layers,
                                  cfg.model.dropout, cfg.model.product)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.base_lr,
                                         weight_decay=cfg.optimizer.weight_decay)
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
        print_logger.info(f"Results for: {cfg.model.type}")
        print_logger.info(f"Embed Model Params: {emb_params}")
        print_logger.info(f"Embed time: {emb_time}")
        print_logger.info(f'Num parameters: {cfg.model.params}')
        trainer.finalize()
        print_logger.info(f"Inference time: {trainer.run_result['eval_time']}")

