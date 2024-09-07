import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.load import load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path
from sentence_transformers import SentenceTransformer
from typing import List
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
import torch.nn as nn

FILE_PATH = f'{get_git_repo_root_path()}/'

args = parse_args()

cfg = set_cfg(FILE_PATH, args.cfg_file)
cfg.merge_from_list(args.opts)

splits, text = load_data_lp[cfg.data.name](cfg.data)


dataset = []
pos_edge_index = splits['train'].pos_edge_label_index
neg_edge_index = splits['train'].neg_edge_label_index

# Process positive edges
for i in range(pos_edge_index.shape[1]):
    node1 = pos_edge_index[0, i].item()
    node2 = pos_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    dataset.append(InputExample(texts=[text1, text2], label=1))

# Process negative edges
for i in range(neg_edge_index.shape[1]):
    node1 = neg_edge_index[0, i].item()
    node2 = neg_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    dataset.append(InputExample(texts=[text1, text2], label=0))

model_name = cfg.embedding_fine_tuning.model_name
model = SentenceTransformer(model_name)
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=16)
train_loss = losses.ContrastiveLoss(model=model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1) 
model_save_path = "./models/"
model.save(model_save_path)