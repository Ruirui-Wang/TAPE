import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import dgl
import torch
import pandas as pd
import numpy as np
import torch
import random
import json
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from graphgps.utility.utils import get_git_repo_root_path  # type: ignore
from typing import Tuple, List, Dict, Set, Any
from data_utils.lcc import use_lcc
import torch_geometric.utils as pyg_utils
import networkx as nx

# import dgl


FILE = 'core/dataset/ogbn_products_orig/ogbn-products.csv'
FILE_PATH = get_git_repo_root_path() + '/'


def get_node_mask(num_nodes: int) -> tuple:
    node_id = torch.randperm(num_nodes)

    train_end = int(num_nodes * 0.6)
    val_end = int(num_nodes * 0.8)

    train_id = torch.sort(node_id[:train_end])[0]
    val_id = torch.sort(node_id[train_end:val_end])[0]
    test_id = torch.sort(node_id[val_end:])[0]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_id] = True

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_id] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_id] = True

    return train_id, val_id, test_id, train_mask, val_mask, test_mask


def get_node_mask_ogb(num_nodes: int, idx_splits: Dict[str, torch.Tensor]) -> tuple:
    train_mask = torch.zeros(num_nodes).bool()
    val_mask = torch.zeros(num_nodes).bool()
    test_mask = torch.zeros(num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    return train_mask, val_mask, test_mask


# Function to parse Cora dataset
def load_graph_arxiv23() -> Data:
    data = torch.load(FILE_PATH + 'core/dataset/arxiv_2023/graph.pt')
    # data.edge_index = data.adj_t.to_symmetric()
    return data


# Function to parse PubMed dataset
def load_text_arxiv23() -> List[str]:
    # Add your implementation here
    df = pd.read_csv(FILE_PATH + 'core/dataset/arxiv_2023_orig/paper_info.csv')
    return [
        f'Title: {ti}\nAbstract: {ab}'
        for ti, ab in zip(df['title'], df['abstract'])
    ]


def load_tag_arxiv23() -> Tuple[Data, List[str]]:
    graph = load_graph_arxiv23()
    text = load_text_arxiv23()

    train_id, val_id, test_id, train_mask, val_mask, test_mask = get_node_mask(graph.num_nodes)
    graph.train_id = train_id
    graph.val_id = val_id
    graph.test_id = test_id
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    return graph, text


def load_graph_cora(use_mask) -> Data:
    path = f'{FILE_PATH}core/dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(f"{path}.content", dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)

    if use_mask:
        labels = idx_features_labels[:, -1]
        class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                                 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning',
                                                 'Theory'])}
        data_Y = np.array([class_map[l] for l in labels])

    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}.cites", dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))

    dataset = Planetoid('./generated_dataset', 'cora',
                        transform=T.NormalizeFeatures())

    x = torch.tensor(data_X).float()
    edge_index = torch.LongTensor(data_edges).T.clone().detach().long()
    num_nodes = len(data_X)

    if use_mask:
        y = torch.tensor(data_Y).long()

    train_id, val_id, test_id, train_mask, val_mask, test_mask = get_node_mask(num_nodes)

    if use_mask:
        return Data(x=x,
                    edge_index=edge_index,
                    y=y,
                    num_nodes=num_nodes,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    val_mask=val_mask,
                    node_attrs=x,
                    edge_attrs=None,
                    graph_attrs=None,
                    train_id=train_id,
                    val_id=val_id,
                    test_id=test_id
                    ), data_citeid

    else:
        return Data(
            x=x,
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_attrs=x,
            edge_attrs=None,
            graph_attrs=None,
        ), data_citeid


def load_tag_cora() -> Tuple[Data, List[str]]:
    data, data_citeid = load_graph_cora(use_mask=False)  # nc True, lp False
    text = load_text_cora(data_citeid)
    print(f"Number of texts: {len(text)}")
    print(f"first text: {text[0]}")
    return data, text


# Function to parse PubMed dataset
def load_text_cora(data_citeid) -> List[str]:
    with open(f'{FILE_PATH}core/dataset/cora_orig/mccallum/cora/papers') as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = f'{FILE_PATH}core/dataset/cora_orig/mccallum/cora/extractions/'

    # for debug
    # save file list
    # with open('extractions.txt', 'w') as txt_file:
    #     # Write each file name to the text file
    #     for file_name in os.listdir(path):
    #         txt_file.write(file_name + '\n')

    text = []
    not_loaded = []
    i = 0
    for pid in data_citeid:
        fn = pid_filename[pid]
        try:
            if os.path.exists(path + fn):
                pathfn = path + fn
            elif os.path.exists(path + fn.replace(":", "_")):
                pathfn = path + fn.replace(":", "_")
            elif os.path.exists(path + fn.replace("_", ":")):
                pathfn = path + fn.replace("_", ":")

            with open(pathfn) as f:
                lines = f.read().splitlines()

            for line in lines:
                if 'Title:' in line:
                    ti = line
                if 'Abstract:' in line:
                    ab = line
            text.append(ti + '\n' + ab)
        except Exception:
            not_loaded.append(pathfn)
            i += 1

    print(f"not loaded {i} papers.")
    return text


# Function to parse PubMed dataset
def load_graph_product():
    raise NotImplementedError
    # Add your implementation here


def load_text_product() -> List[str]:
    text = pd.read_csv(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n' for ti,
    cont in zip(text['title'], text['content'])]
    return text


# Function to parse PubMed dataset
def load_tag_product() -> Tuple[Data, List[str]]:
    data = torch.load(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n' for ti,
    cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    return data, text


def parse_pubmed():
    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(FILE_PATH + 'core/dataset/PubMed_orig/data/Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                    1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(FILE_PATH + 'core/dataset/PubMed_orig/data/Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()


def load_graph_pubmed(use_mask) -> Data:
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./generated_dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    x = torch.tensor(data_X)
    edge_index = torch.tensor(data_edges)
    num_nodes = data.num_nodes

    # split data
    if use_mask:
        y = torch.tensor(data_Y)
        train_id, val_id, test_id, train_mask, val_mask, test_mask = get_node_mask(num_nodes)

        return Data(x=x,
                    edge_index=edge_index,
                    y=y,
                    num_nodes=num_nodes,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    val_mask=val_mask,
                    node_attrs=x,
                    edge_attrs=None,
                    graph_attrs=None,
                    train_id=train_id,
                    val_id=val_id,
                    test_id=test_id
                    )
    else:
        return Data(x=x,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    node_attrs=x,
                    edge_attrs=None,
                    graph_attrs=None
                    )


# Function to parse PubMed dataset
def load_text_pubmed() -> List[str]:
    f = open(FILE_PATH + 'core/dataset/PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    return ['Title: ' + ti + '\n' + 'Abstract: ' + ab for ti, ab in zip(TI, AB)]


def load_tag_pubmed(use_mask) -> Tuple[Data, List[str]]:
    # Add your implementation here
    graph = load_graph_pubmed(use_mask)
    text = load_text_pubmed()
    return graph, text


def load_text_ogbn_arxiv():
    nodeidx2paperid = pd.read_csv(
        FILE_PATH + 'core/dataset/ogbn_arixv_orig/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    tsv_path = FILE_PATH + 'core/dataset/ogbn_arixv_orig/titleabs.tsv'
    raw_text = pd.read_csv(tsv_path,
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    raw_text['paper id'] = pd.to_numeric(raw_text['paper id'], errors='coerce')
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    return [
        'Title: ' + ti + '\n' + 'Abstract: ' + ab
        for ti, ab in zip(df['title'], df['abs'])
    ]


def load_graph_ogbn_arxiv(use_mask):
    dataset = PygNodePropPredDataset(root='./generated_dataset',
                                     name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]

    if data.adj_t.is_symmetric():
        is_symmetric = True
    else:
        edge_index = data.adj_t.to_symmetric()

    x = torch.tensor(data.x).float()
    edge_index = torch.LongTensor(edge_index.to_torch_sparse_coo_tensor().coalesce().indices()).long()
    num_nodes = data.num_nodes

    if use_mask:
        y = torch.tensor(data.y).long()
        train_mask, val_mask, test_mask = get_node_mask_ogb(data.num_nodes, dataset.get_idx_split())

        return Data(x=x,
                    edge_index=edge_index,
                    y=y,
                    num_nodes=num_nodes,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    val_mask=val_mask,
                    node_attrs=x,
                    edge_attrs=None,
                    graph_attrs=None
                    )

    else:
        return Data(x=x,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    node_attrs=x,
                    edge_attrs=None,
                    graph_attrs=None
                    )


def load_tag_ogbn_arxiv() -> List[str]:
    graph = load_graph_ogbn_arxiv(False)
    text = load_text_ogbn_arxiv()
    return graph, text


def load_tag_product() -> Tuple[Data, List[str]]:
    data = torch.load(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n' for ti,
    cont in zip(text['title'], text['content'])]

    edge_index = data.adj_t.to_symmetric().to_torch_sparse_coo_tensor().coalesce().indices()
    data.edge_index = torch.LongTensor(edge_index).long()

    return data, text


def load_graph_citationv8() -> Data:
    import dgl
    from pdb import set_trace as st;
    st()
    graph = dgl.load_graphs(FILE_PATH + 'core/dataset/citationv8/Citation-2015.pt')[0][0]
    graph = dgl.to_bidirected(graph)
    from torch_geometric.utils import from_dgl
    graph = from_dgl(graph)
    graph.num_nodes = graph.edge_index.max() + 1
    # torch.save(graph, FILE_PATH + 'core/dataset/citationv8/citationv8_pyg2015.pt')
    return graph


def load_pyg_citationv8() -> Data:
    return torch.load(FILE_PATH + 'core/dataset/citationv8/citationv8_pyg2015.pt')


def load_embedded_citationv8(method) -> Data:
    return torch.load(FILE_PATH + f'core/dataset/citationv8/citationv8_{method}.pt')


def load_text_citationv8() -> List[str]:
    df = pd.read_csv(FILE_PATH + 'core/dataset/citationv8_orig/Citation-2015.csv')
    print(f"Number of texts: {len(df['text'].tolist())}")
    return df['text'].tolist()


def load_tag_citationv8() -> Tuple[Data, List[str]]:
    graph = load_graph_citationv8()
    text = None
    train_id, val_id, test_id, train_mask, val_mask, test_mask = get_node_mask(graph.num_nodes)
    graph.train_id = train_id
    graph.val_id = val_id
    graph.test_id = test_id
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    return graph, text


def load_graph_citeseer() -> Data:
    # load data
    data_name = 'CiteSeer'
    dataset = Planetoid('./generated_dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    return data


def load_text_citeseer() -> List[str]:
    return None


def load_tag_citeseer() -> Tuple[Data, List[str]]:
    graph = load_graph_citeseer()
    text = load_text_citeseer()
    return graph, text


def load_graph_pwc_large(method):
    graph = torch.load(FILE_PATH + f'core/dataset/pwc_large/pwc_{method}_large_undirec.pt')
    return graph


def load_text_pwc_large() -> List[str]:
    raw_text = pd.read_csv(FILE_PATH + f'core/dataset/pwc_large/pwc_large_papers.csv')
    return raw_text['feat'].tolist()


def load_graph_pwc_medium(method):
    return torch.load(FILE_PATH + f'core/dataset/pwc_medium/pwc_{method}_medium_undirec.pt')


def load_text_pwc_medium(method) -> List[str]:
    raw_text = pd.read_csv(FILE_PATH + f'core/dataset/pwc_medium/pwc_{method}_medium_text.csv')
    return raw_text['feat'].tolist()


def load_graph_pwc_small(method):
    return torch.load(FILE_PATH + f'core/dataset/pwc_small/pwc_{method}_small_undirec.pt')


def load_text_pwc_small(method) -> List[str]:
    raw_text = pd.read_csv(FILE_PATH + f'core/dataset/pwc_small/pwc_{method}_small_text.csv')
    return raw_text['feat'].tolist()


def extract_lcc_pwc_undir() -> Data:
    # return the largest connected components with text attrs
    graph = torch.load(FILE_PATH + 'core/dataset/pwc_large/pwc_tfidf_large_undir.pt')
    data_lcc = use_lcc(graph)
    root = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/'
    torch.save(data_lcc, root + 'core/dataset/pwc_large/pwc_tfidf_medium_undir.pt')
    from pdb import set_trace as st;
    st()
    return


# Test code
if __name__ == '__main__':
    graph = load_graph_citeseer()
    print(type(graph))
    graph, text = load_tag_citeseer()
    print(type(text))

    graph = load_graph_arxiv23()
    # print(type(graph))
    graph, text = load_tag_arxiv23()
    print(type(graph))
    print(type(text))

    '''graph, _ = load_graph_cora(True)
    # print(type(graph))
    graph, text = load_tag_cora()
    print(type(graph))
    print(type(text))

    graph, text = load_tag_ogbn_arxiv()
    print(type(graph))
    print(type(text))

    graph, text = load_tag_product()
    print(type(graph))
    print(type(text))

    graph = load_graph_pubmed()
    graph, text = load_tag_pubmed()
    print(type(graph))
    print(type(text))

    graph = load_graph_citationv8()
    print(type(graph))
    graph, text = load_tag_citationv8()
    print(type(text))'''