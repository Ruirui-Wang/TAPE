import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
from typing import Dict
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from yacs.config import CfgNode as CN
from typing import Dict, Tuple, List, Union
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from tqdm import tqdm 
import time

from data_utils.load_data_nc import load_tag_cora, load_tag_pubmed, \
    load_tag_product, load_tag_ogbn_arxiv, load_tag_product, \
    load_tag_arxiv23, load_graph_cora, load_graph_pubmed, \
    load_graph_arxiv23, load_graph_ogbn_arxiv, load_text_cora, \
    load_text_pubmed, load_text_arxiv23, load_text_ogbn_arxiv, \
    load_text_product, load_text_citeseer, load_text_citationv8, \
    load_graph_citeseer, load_graph_citationv8, load_graph_pwc_large, load_text_pwc_large, \
    load_graph_pwc_medium, load_text_pwc_medium, load_text_pwc_small,  load_graph_pwc_small, \
    load_embedded_citationv8, load_pyg_citationv8, load_tag_photo, load_tag_history
from graphgps.utility.utils import get_git_repo_root_path, config_device, init_cfg_test
from data_utils.lcc import find_scc_direc, use_lcc_direc, use_lcc


FILE = 'core/dataset/ogbn_products_orig/ogbn-products.csv'
FILE_PATH = get_git_repo_root_path() + '/'


# arxiv_2023
def load_taglp_arxiv2023(cfg: CN, if_lcc: bool=True, alg_name: str='', node_features=None) -> Tuple[Dict[str, Data], List[str]]:

    data, text = load_tag_arxiv23()
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    print(f"original num of nodes: {data.num_nodes}")
    
    if if_lcc:
        data, lcc, _ = use_lcc(data)
        text = [text[i] for i in lcc]
    
    if node_features is not None:
        data.x = node_features
        
    if data.is_directed() is True:
        data.edge_index = to_undirected(data.edge_index)
        undirected = True
    
    if alg_name.lower() == 'hl-gnn':
        return [], [], data
    else:
        splits = get_edge_split(data,
                                undirected,
                                cfg.device,
                                cfg.split_index[1],
                                cfg.split_index[2],
                                cfg.include_negatives,
                                cfg.split_labels
                                )
        print(f"num of nodes after lcc: {data.num_nodes}")
        print(f"num of edges after lcc: {data.edge_index.shape[1]}")
        print(f"num of texts in dataset: {len(text)}")
        return splits, text, data


def load_taglp_cora(cfg: CN, if_lcc: bool=True, alg_name: str='', node_features=None) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, data_citeid = load_graph_cora(False)
    text = load_text_cora(data_citeid)
    
    if if_lcc: 
        data, lcc, _ = use_lcc(data)
        
        text = [text[i] for i in lcc]
    
    if node_features is not None:
        data.x = node_features
    
    edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    if torch.__version__ == '2.2.1':
        edge_index = remove_self_loops(edge_index)[0]
    
    data.edge_index = edge_index
    print(f"original num of nodes: {data.num_nodes}")
    if alg_name.lower() == 'hl-gnn':
        return [], [], data
    else:
        undirected = data.is_undirected()

        splits = get_edge_split(data,
                                undirected,
                                cfg.device,
                                cfg.split_index[1],
                                cfg.split_index[2],
                                cfg.include_negatives,
                                cfg.split_labels
                                )
        print(f"num of nodes after lcc: {data.num_nodes}")
        print(f"num of edges after lcc: {data.edge_index.shape[1]}")
        print(f"num of texts in dataset: {len(text)}")
        
        return splits, text, data


def load_taglp_ogbn_arxiv(cfg: CN, if_lcc) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_ogbn_arxiv(False)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    text = load_text_ogbn_arxiv()
    undirected = data.is_undirected()

    print(f"original num of nodes: {data.num_nodes}")
    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    print(f"num of nodes after lcc: {data.num_nodes}")
    print(f"num of edges after lcc: {data.edge_index.shape[1]}")
    print(f"num of texts in dataset: {len(text)}")
    return splits, text, data

def load_taglp_pwc_large(cfg: CN, if_lcc) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_pwc_large(cfg.method)
    text = load_text_pwc_large()
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    undirected = data.is_undirected()

    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def get_edge_split(data: Data,
                   undirected: bool,
                   device: Union[str, int],
                   val_pct: float,
                   test_pct: float,
                   include_negatives: bool,
                   split_labels: bool):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomLinkSplit(is_undirected=undirected,
                        num_val=val_pct,
                        num_test=test_pct,
                        add_negative_train_samples=include_negatives,
                        split_labels=split_labels),

    ])
    del data.adj_t, data.e_id, data.batch_size, data.n_asin, data.n_id
    del data.full_adj_t, data.max_x
    train_data, val_data, test_data = transform(data)
    return {'train': train_data, 'valid': val_data, 'test': test_data}


def load_taglp_product(cfg: CN, if_lcc) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, text = load_tag_product()
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    undirected = data.is_undirected()


    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.5f} seconds")
        return result
    return wrapper

@time_function
def load_taglp_pubmed(cfg: CN, if_lcc: bool=True, alg_name: str='', node_features=None) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    # I added this condition for planetoid, later I will delete
    data = load_graph_pubmed(False)
    text = load_text_pubmed()
    if if_lcc: 
        data, lcc, _ = use_lcc(data)
        text = [text[i] for i in lcc]
    
    if node_features is not None:
        data.x = node_features
           
    if alg_name.lower() == 'hl-gnn':
        return [], [], data
    else:
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
        data.edge_index, _ = remove_self_loops(data.edge_index)
        undirected = data.is_undirected()

        splits = get_edge_split(data,
                                undirected,
                                cfg.device,
                                cfg.split_index[1],
                                cfg.split_index[2],
                                cfg.include_negatives,
                                cfg.split_labels
                                )
        return splits, text, data
        

def load_taglp_citeseer(cfg: CN, if_lcc) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_citeseer()
    text = load_text_citeseer()
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    undirected = data.is_undirected()

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data

def load_taglp_citationv8(cfg: CN, lcc_bool: bool=True) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument
    
    data = load_embedded_citationv8('tfidf')
    text = load_text_citationv8()
    
    print(f"original num of nodes: {data.num_nodes}")
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
    
    if lcc_bool:
        data, lcc, _ = use_lcc(data)
        text = [text[i] for i in lcc]
    
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    print(f"num of nodes after lcc: {data.num_nodes}")
    print(f"num of edges after lcc: {data.edge_index.shape[1]}")
    print(f"num of texts in dataset: {len(text)}")
    print(f"split_train edges: {splits['train'].edge_index.max().tolist()+1}")
    print(f"split_valid edges: {splits['valid'].edge_index.max().tolist()+1}")
    print(f"split_test edges: {splits['test'].edge_index.max().tolist()+1}")
    return splits, text, data


def load_taglp_photo(cfg: CN, lcc_bool: bool = True) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, text = load_tag_photo()

    print(f"original num of nodes: {data.num_nodes}")
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if data.is_directed() is True:
        data.edge_index = to_undirected(data.edge_index)
        undirected = True
    else:
        undirected = data.is_undirected()

    if lcc_bool:
        data, lcc, _ = use_lcc(data)
        text = [text[i] for i in lcc]

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    print(f"num of nodes after lcc: {data.num_nodes}")
    print(f"num of edges after lcc: {data.edge_index.shape[1]}")
    print(f"num of texts in dataset: {len(text)}")
    print(f"split_train edges: {splits['train'].edge_index.max().tolist() + 1}")
    print(f"split_valid edges: {splits['valid'].edge_index.max().tolist() + 1}")
    print(f"split_test edges: {splits['test'].edge_index.max().tolist() + 1}")
    return splits, text, data

def load_taglp_history(cfg: CN, lcc_bool: bool = True) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, text = load_tag_history()

    print(f"original num of nodes: {data.num_nodes}")
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if data.is_directed() is True:
        data.edge_index = to_undirected(data.edge_index)
        undirected = True
    else:
        undirected = data.is_undirected()

    if lcc_bool:
        data, lcc, _ = use_lcc(data)
        text = [text[i] for i in lcc]

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    print(f"num of nodes after lcc: {data.num_nodes}")
    print(f"num of edges after lcc: {data.edge_index.shape[1]}")
    print(f"num of texts in dataset: {len(text)}")
    print(f"split_train edges: {splits['train'].edge_index.max().tolist() + 1}")
    print(f"split_valid edges: {splits['valid'].edge_index.max().tolist() + 1}")
    print(f"split_test edges: {splits['test'].edge_index.max().tolist() + 1}")
    return splits, text, data

def load_taglp_pwc_large(cfg: CN, if_lcc) -> Tuple[Dict[str, Data], List[str]]:
    if hasattr(cfg, 'method'):
        pass
    else:
        cfg.method = 'w2v'
    data = load_graph_pwc_large(cfg.method)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    df, text = load_text_pwc_large()
    
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
        
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, df, data


def load_taglp_pwc_medium(cfg: CN, if_lcc) -> Tuple[Dict[str, Data], List[str]]:
    if hasattr(cfg, 'method'):
        pass
    else:
        cfg.method = 'w2v'
    data = load_graph_pwc_medium(cfg.method)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    text = load_text_pwc_medium(cfg.method)
    print(f"original num of nodes: {data.num_nodes}")
    
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
    
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    print(f"num of nodes after lcc: {data.num_nodes}")
    print(f"num of edges after lcc: {data.edge_index.shape[1]}")
    print(f"num of texts in dataset: {len(text)}")
    return splits, text, data


def load_taglp_pwc_small(cfg: CN, if_lcc = True) -> Tuple[Dict[str, Data], List[str]]:
    if hasattr(cfg, 'method'):
        pass
    else:
        cfg.method = 'w2v'
    data = load_graph_pwc_small(cfg.method)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    
    print(f"original num of nodes: {data.num_nodes}")
    text = load_text_pwc_small(cfg.method)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
    

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    print(f"num of nodes after lcc: {data.num_nodes}")
    print(f"num of edges after lcc: {data.edge_index.shape[1]}")
    print(f"num of texts in dataset: {len(text)}")
    return splits, text, data


def preprocess(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)


# Function to get the average embedding for a whole text (e.g., title and abstract combined)
def get_average_embedding(text, model):
    tokens = preprocess(text)
    embeddings = [model.wv[token] for token in tokens if token in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        # Return a zero vector if none of the tokens are in the vocabulary
        return np.zeros(model.vector_size)


def load_text_benchmark(data_name: str) -> pd.DataFrame:
    if  data_name == 'pwc_small':
        df = load_text_pwc_small('tfidf')
    if data_name == 'cora':
        _, df = load_tag_cora()
    if data_name == 'pubmed':
        df = load_text_pubmed()
    if data_name == 'arxiv_2023':
        df = load_text_arxiv23()
    if data_name == 'pwc_medium':
        df = load_text_pwc_medium('tfidf')
    if data_name == 'ogbn-arxiv':
        df = load_text_ogbn_arxiv()
    if data_name == 'citationv8':
        df = load_text_citationv8()
    if data_name == 'pwc_large':
        df = load_text_pwc_large()
    if type(df) is list:
        df = pd.DataFrame(df, columns=['text'])
        return df


def token_statistic(datasets):
    all_stats_df = []
    
    for data_name in datasets:
        # Load dataset
        df = load_text_benchmark(data_name)
        
        # Ensure nltk tokenizers are downloaded
        nltk.download('punkt')
        
        # If df is a list, convert it to a DataFrame^
        if isinstance(df, list):
            df = pd.DataFrame(df, columns=['text'])
        
        # Tokenize the node features
        df['tokens'] = df['text'].apply(word_tokenize)
        df['size_in_bytes'] = df['text'].apply(lambda x: len(x.encode('utf-8')))
        total_size_in_bytes = df['size_in_bytes'].sum()
        total_size_in_megabytes = total_size_in_bytes / (1024 * 1024)
        
        
        # Count the number of tokens for each node
        df['num_tokens'] = df['tokens'].apply(len)
        
        # Provide statistical analysis
        total_tokens = df['num_tokens'].sum()
        average_tokens_per_node = df['num_tokens'].mean()
        token_count_distribution = df['num_tokens'].describe()
        
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per node: {average_tokens_per_node}")
        print("Token count distribution:")
        print(token_count_distribution)
        
        # Create a dictionary to store the statistics
        stats = {
            'data_name': data_name,
            'total_tokens': total_tokens,
            'average_tokens_per_node': average_tokens_per_node,
            'count': token_count_distribution['count'],
            'mean': token_count_distribution['mean'],
            'std': token_count_distribution['std'],
            'min': token_count_distribution['min'],
            '25%': token_count_distribution['25%'],
            '50%': token_count_distribution['50%'],
            '75%': token_count_distribution['75%'],
            'max': token_count_distribution['max'],
            'data size': total_size_in_megabytes
        }
        
        # Append the statistics to the all_stats_df DataFrame
        all_stats_df.append(stats)
        
        # Plot the distribution of token counts
        plt.figure(figsize=(10, 6))
        sns.histplot(df['num_tokens'], kde=True, bins=30)
        plt.title(f'Distribution of Token Counts for {data_name}')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        plt.savefig(f'{data_name}.png')

    # Save the all_stats_df DataFrame to a CSV file
    all_stats_df = pd.DataFrame(all_stats_df)
    all_stats_df.to_csv('all_datasets_statistics.csv', index=False)

    print("All statistics have been saved to 'all_datasets_statistics.csv'")


# TEST CODE
if __name__ == '__main__':
    
    args = init_cfg_test()
    args = config_device(args)

    # List of datasets to process
    datasets = ['pwc_small', 'cora', 'pubmed', 'arxiv_2023', 'pwc_medium', 'pwc_large']

    token_statistic(datasets)
    exit(-1)
    from pdb import set_trace as st; st()
    data = load_embedded_citationv8(args.data.method)
    print(data)
    
    preprocessed_texts = [preprocess(t[0]) for t in tqdm(text)]
    print(len(preprocessed_texts))
    # Train a Word2Vec model
    
    model = Word2Vec(sentences=preprocessed_texts, vector_size=128, window=5, min_count=1, workers=10)

    w2v_nodefeat = np.array([get_average_embedding(t[0], model) for t in text])
    
    x = torch.tensor(w2v_nodefeat, dtype=torch.float)
    
    data.x = x
    torch.save(data, f'citationv8_{args.data.method}.pt')
    exit(-1)
    vectorizer = TfidfVectorizer(max_features=128)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    from pdb import set_trace as st; st()
    x = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float)
    
    data.x = x
    torch.save(data, f'citationv8_{args.data.method}.pt')
    exit(-1)
    from core.data_utils.lcc import use_lcc
    splits, text, data = load_taglp_pwc_small(args.data)
    print(splits)
    print(text.iloc[0])
    print(data)
    splits, text, data = load_taglp_pwc_medium(args.data)
    print(splits)
    print(text.iloc[0])
    print(data)
    splits, text, data = load_taglp_pwc_large(args.data)
    print(splits)
    print(text.iloc[0])
    print(data)
    exit(-1)
    # return the largest connected components with text attrs
    graph = torch.load(FILE_PATH+f'core/dataset/pwc_large/pwc_{method}_large_undirec.pt')
    df, text = load_text_pwc_large()
    
    data_lcc, lcc = use_lcc(graph)
    root = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/'
    torch.save(data_lcc, root+f'core/dataset/pwc_medium/pwc_{method}_medium_undirec.pt')
    df_lcc = df.iloc[lcc.tolist()]
    df_lcc.to_csv(root+f'core/dataset/pwc_medium/pwc_{method}_medium_text.csv')

    graph = torch.load(FILE_PATH + f'core/dataset/pwc_large/pwc_{method}_large_direc.pt')
    
    largest_scc = find_scc_direc(graph)
    lcc = list(largest_scc)
    print("Nodes in the largest strongly connected component:", len(lcc))
    df_lcc_direc = df.iloc[lcc]
    df_lcc_direc.to_csv(root+f'core/dataset/pwc_small/pwc_{method}_small_text.csv')
    subgraph = use_lcc_direc(graph, largest_scc)
    torch.save(subgraph, root+f'core/dataset/pwc_small/pwc_{method}_small_undirec.pt')
    exit(-1)

    print('pwc_large')
    print(args.data)
    splits, text, data = load_taglp_pwc_large(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")

    root = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/'
    path_large_tfidf_undir = root + 'core/dataset/pwc_large/pwc_tfidf_large_undirec.pt'
    path_large_tfidf_dir = root + 'core/dataset/pwc_large/pwc_tfidf_large_direc.pt'

    path_large_w2v_undir = root + 'core/dataset/pwc_large/pwc_w2v_large_undirec.pt'
    path_large_w2v_dir = root + 'core/dataset/pwc_large/pwc_w2v_large_direc.pt'
 
    path_medium_tfidf_undir = root + 'core/dataset/pwc_medium/pwc_tfidf_medium_undirec.pt'
    path_medium_w2v_undir = root + 'core/dataset/pwc_medium/pwc_w2v_medium_undirec.pt'
   
    path_small_tfidf_undir = root + 'core/dataset/pwc_small/pwc_tfidf_small_undirec.pt'
    path_small_w2v_undir = root + 'core/dataset/pwc_small/pwc_w2v_small_undirec.pt'
     
    graph = torch.load(path_large_tfidf_undir)
    print(f'directed: {graph.is_directed()}')
    
    graph = torch.load(path_large_tfidf_dir)
    print(f'directed: {graph.is_directed()}')
    
    graph = torch.load(path_large_w2v_undir)
    print(f'directed: {graph.is_directed()}')
    
    graph = torch.load(path_large_w2v_dir)
    print(f'directed: {graph.is_directed()}')
    
    graph = torch.load(path_medium_tfidf_undir)
    print(f'directed: {graph.is_directed()}')
    
    graph = torch.load(path_medium_w2v_undir)
    print(f'directed: {graph.is_directed()}')
    
    graph = torch.load(path_small_tfidf_undir)
    print(f'directed: {graph.is_directed()}')
    
    graph = torch.load(path_small_w2v_undir)
    print(f'directed: {graph.is_directed()}')
    
    print('arxiv2023')
    splits, text, data  = load_taglp_arxiv2023(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")
    
    print('citationv8')
    splits, text, data = load_taglp_citationv8(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")

    print('cora')
    splits, text, data = load_taglp_cora(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")
    
    print('pubmed')
    splits, text, data = load_taglp_pubmed(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")


    
    print(args.data)
    splits, text, data = load_taglp_ogbn_arxiv(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")


    # print('product')
    # splits, text, data = load_taglp_product(args.data)
    # print(f'directed: {data.is_directed()}')
    # print(data)
    # print(text[0])
    # print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    # print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    # print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")

    # splits, text, data = load_taglp_citeseer(args.data)
    # print(f'directed: {data.is_directed()}')
    # print(data)
    # # print(text[0])
    # print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    # print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    # print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")