import copy
import os, sys
import gc
import transformers
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from torch import Tensor, nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

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
from graphgps.train.embedding_LLM_train import Trainer_embedding_LLM
from graphgps.utility.utils import save_run_results_to_csv, random_sampling
from graphgps.utility.utils import random_sampling
from graphgps.score.custom_score import LinkPredictor
from data_utils.remap_and_visualize import remap_node_indices


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/lms/minilm.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', type=str, required=False, default='ogbn-arxiv',
                        help='data name')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    parser.add_argument('--device', dest='device', required=False,
                        help='device id')
    parser.add_argument('--downsampling', type=float, default=1,
                        help='Downsampling rate.')
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
    args.data = cfg.data.name

    torch.set_num_threads(cfg.num_threads)
    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    cfg.device = args.device
    splits, text, data = load_data_lp[cfg.data.name](cfg.data)
    data, splits = remap_node_indices(data, splits)
    splits = random_sampling(splits, args.downsampling)

    saved_features_path = './' + cfg.embedder.type + cfg.data.name + 'saved_node_features.pt'
    if os.path.exists(saved_features_path):
        node_features = torch.load(saved_features_path)
    else:
        if cfg.embedder.type == 'minilm':
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=cfg.device)
            node_features = model.encode(text, batch_size=256)
        elif cfg.embedder.type == 'e5-large':
            model = SentenceTransformer('intfloat/e5-large-v2', device=cfg.device)
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
            node_features = []
            batch_size = 256
            for i in range(0, len(text), batch_size):
                batch_texts = text[i:i + batch_size]
                encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                          max_length=512).to(cfg.device)
                with torch.no_grad():
                    outputs = model(**encoded_input)
                    batch_features = outputs.pooler_output
                    node_features.append(batch_features)
            node_features = torch.cat(node_features, dim=0)
        torch.save(node_features, saved_features_path)
    node_features = torch.tensor(node_features)
    print(node_features.shape)

    '''tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(node_features)
    print('tsne done')
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(node_features)
    print('pca done')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1])
    plt.title("t-SNE")
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1])
    plt.title("PCA")
    plt.show()'''
    import plotly.express as px
    import plotly.graph_objects as go

    heatmap_data = go.Heatmap(z=node_features, colorscale='Viridis')

    layout = go.Layout(
        title="Feature Visualization",
        xaxis_title="Feature Dimension",
        yaxis_title="Node Index"
    )
    fig = go.Figure(data=[heatmap_data], layout=layout)
    fig.show()