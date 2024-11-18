import os
import sys
sys.path.insert(0, '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core')
import copy as cp
import numpy as np

import torch 
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as ssp
from data_utils.load_data_nc import (
    load_graph_cora, 
    load_graph_pubmed, 
    load_tag_arxiv23, 
    load_graph_ogbn_arxiv
)
from graphgps.encoder.seal import do_edge_split, do_ogb_edge_split
from torch_geometric.utils import to_scipy_sparse_matrix
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import networkx 
if networkx.__version__ == '2.6.3':
    from networkx import from_scipy_sparse_matrix as from_scipy_sparse_array
else:
    from networkx import from_scipy_sparse_array

from torch_geometric.loader import DataLoader
from tqdm import tqdm 

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_graph(dataset: str):
    if dataset == 'custom-pubmed':
        data = load_graph_pubmed(False)
    elif dataset == 'custom-cora':
        data, _ = load_graph_cora(False)
    elif dataset == 'custom-arxiv_2023':
        data, _ = load_tag_arxiv23()
    elif dataset == 'custom-ogbn-arxiv':
        data = load_graph_ogbn-arxiv(False)
    
    if dataset.startswith('custom'):
        split_edge = do_ogb_edge_split(cp.deepcopy(data), fast_split)
        return data, split_edge
    
    if dataset.startswith('ogbl'):
        data = PygLinkPropPredDataset(name=dataset)
        split_edge = data.get_edge_split()
        data = data[0]
        if dataset.startswith('ogbl-vessel'):
            # normalize node features
            data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
            data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
            data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)   
    else:
        path = osp.join('generated_dataset', dataset)
        data = Planetoid(path, dataset)
        data = data[0]
        split_edge = do_ogb_edge_split(data, fast_split)
        data.edge_index = split_edge['train']['edge'].t()
    return data, split_edge

def plot_degree_distribution(G, color, dataset):
    
    # Get the degree of each node
    degrees = [G.degree(n) for n in G.nodes()]

    # Compute degree distribution
    degree_counts = np.bincount(degrees)
    degree_distribution = degree_counts / len(G.nodes())

    # Compute cumulative degree distribution
    cumulative_distribution = np.cumsum(degree_distribution[::-1])[::-1]

    # Individual plot for each dataset

    plt.loglog(range(len(cumulative_distribution)), cumulative_distribution, marker='o', linestyle='-', color=color, label=dataset)


def plot_clustering_specturm(G, color, dataset):

    clustering_coeffs = nx.clustering(G)

    # Extract the clustering coefficient values
    coeff_values = list(clustering_coeffs.values())
    cluster_coef = nx.average_clustering(G)
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(coeff_values, bins=30, edgecolor='k', alpha=0.7)
    plt.title('Clustering Coefficient Distribution avg_clustering: {:.4f}'.format(cluster_coef))
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{dataset}_clustering_distribution.pdf')


def get_test_hist(data_name, pos_test_pred, neg_test_pred, use_heuristic):
   
    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    
    pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist, bin_edges = np.histogram(pred, bins=bin_edges)
    
    sns.barplot(x=[1, 2, 3, 4, 5], y=hist, color='skyblue')

    


def CN(A, edge_index, batch_size=100000):
    """
    Common neighbours
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    edge_index = edge_index.t()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size, num_workers=4)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


from typing import Dict
def CN_citation2(A, edge_index: Dict, batch_size=100000):
    """
    Common neighbours
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    

    link_loader = DataLoader(range(edge_index['source_node'].shape[0]), batch_size, num_workers=4)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index['source_node'][ind], edge_index['target_node'][ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index



   
def shortest_path(G, edge_index, remove=False):

    scores = []
    count = 0
    print('remove: ', remove)
    edge_index = edge_index.t()
    for i in tqdm(range(edge_index.size(1))):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        if nx.has_path(G, source=s, target=t):
            sp = nx.shortest_path_length(G, source=s, target=t)
        else:
            sp = 999
        # scores.append(1/(sp))
        scores.append(np.exp(-(sp-1)))
    print(f'evaluated shortest path for {scores[:20]} edges')
    return torch.FloatTensor(scores)


def shortest_path_citation2(G, edge_index, remove=False):
    scores = []
    count = 0
    print('remove: ', remove)
    
    for i in tqdm(range(edge_index['source_node'].size(0))):
        
        s = edge_index['source_node'][i].item()
        t = edge_index['target_node'][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        if nx.has_path(G, source=s, target=t):
            sp = nx.shortest_path_length(G, source=s, target=t)
        else:
            sp = 999
        # scores.append(1/(sp))
        scores.append(np.exp(-(sp-1)))
    print(f'evaluated shortest path for {scores[:20]} edges')
    return torch.FloatTensor(scores)


def plot_CN_dist(A, split_edge, dataset):
    use_heuristic = 'CN'
    pos_train_pred, _ = CN(A, split_edge['train']['edge'])

    neg_edge_index = negative_sampling(
    data.edge_index, num_nodes=num_nodes,
    num_neg_samples=pos_train_pred.size(0))
    neg_train_pred, _ = CN(A, neg_edge_index)

    pos_valid_pred, _ = CN(A, split_edge['valid']['edge'])
    neg_valid_pred, _ = CN(A, split_edge['valid']['edge_neg'])
   
    pos_test_pred, _ = CN(A, split_edge['test']['edge'])
    neg_test_pred, _ = CN(A, split_edge['test']['edge_neg'])

    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    colors = sns.color_palette("husl", 3)  # 'husl' is one of many color palettes available

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist_test, bin_edges = np.histogram(test_pred, bins=bin_edges)
    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    hist_train, bin_edges = np.histogram(train_pred, bins=bin_edges)
    valid_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    hist_valid, bin_edges = np.histogram(valid_pred, bins=bin_edges)
    colors = sns.color_palette("husl", 3)

    bar_width = 0.25
    indices = np.arange(len(hist_train))
    hist_train = hist_train / hist_train.max()
    hist_valid = hist_valid / hist_valid.max()
    hist_test = hist_test / hist_test.max()
    
    plt.figure(figsize=(8, 6))
    plt.bar(indices, hist_train, bar_width, color=colors[0], alpha=0.7, label='Train')
    plt.bar(indices + bar_width, hist_valid, bar_width, color=colors[1], alpha=0.7, label='Valid')
    plt.bar(indices + 2 * bar_width, hist_test, bar_width, color=colors[2], alpha=0.7, label='Test')

    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} CN Distribution', fontsize=24)
    plt.xlabel('Num of CN', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to {dataset}_{use_heuristic}_distribution.pdf')
    
    plt.figure(figsize=(8, 6))
    hist_test_pos, bin_edges = np.histogram(pos_test_pred, bins=bin_edges)
    hist_test_neg, bin_edges = np.histogram(neg_test_pred, bins=bin_edges)
    plt.bar(indices, hist_test_pos, color='green', width=bar_width, edgecolor='grey', label='Positive', hatch='/')
    # Negative samples
    plt.bar(indices + bar_width, hist_test_neg, color='blue', width=bar_width, edgecolor='grey', label='Negative', hatch='\\')
    
    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} CN Distribution', fontsize=24)
    plt.xlabel('Num of CN', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'pos_neg_{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to pos_neg_{dataset}_{use_heuristic}_test_distribution.pdf')
    

def plot_CN_citation2_dist(A, split_edge, dataset):
    
    use_heuristic = 'CN'
    pos_train_pred, _ = CN_citation2(A, split_edge['train'])
    # create edge_index from source and target node
    edge_index  = torch.stack([split_edge['train']['source_node'], split_edge['train']['target_node']])

    neg_edge_index = negative_sampling(
    edge_index, num_nodes=num_nodes,
    num_neg_samples=pos_train_pred.size(0))
    neg_train_index = {
        'source_node': neg_edge_index[0],
        'target_node': neg_edge_index[1],
    }
    neg_train_pred, _ = CN_citation2(A, neg_train_index)

    pos_valid_pred, _ = CN_citation2(A, split_edge['valid'])
    neg_valid_pred, _ = CN_citation2(A, split_edge['valid'])
    
    pos_test_pred, _ = CN_citation2(A, split_edge['test'])
    neg_test_pred, _ = CN_citation2(A, split_edge['test'])
   
    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    colors = sns.color_palette("husl", 3)  # 'husl' is one of many color palettes available

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist_test, bin_edges = np.histogram(test_pred, bins=bin_edges)
    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    hist_train, bin_edges = np.histogram(train_pred, bins=bin_edges)
    valid_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    hist_valid, bin_edges = np.histogram(valid_pred, bins=bin_edges)
    colors = sns.color_palette("husl", 3)

    bar_width = 0.25
    indices = np.arange(len(hist_train))
    hist_train = hist_train / hist_train.max()
    hist_valid = hist_valid / hist_valid.max()
    hist_test = hist_test / hist_test.max()
    plt.figure(figsize=(8, 6))
    plt.bar(indices, hist_train, bar_width, color=colors[0], alpha=0.7, label='Train')
    plt.bar(indices + bar_width, hist_valid, bar_width, color=colors[1], alpha=0.7, label='Valid')
    plt.bar(indices + 2 * bar_width, hist_test, bar_width, color=colors[2], alpha=0.7, label='Test')

    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} CN Distribution', fontsize=24)
    plt.xlabel('Num of CN', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to {dataset}_{use_heuristic}_distribution.pdf')

    plt.figure(figsize=(8, 6))
    hist_test_pos, bin_edges = np.histogram(pos_test_pred, bins=bin_edges)
    hist_test_neg, bin_edges = np.histogram(neg_test_pred, bins=bin_edges)
    plt.bar(indices, hist_test_pos, color='green', width=bar_width, edgecolor='grey', label='Positive', hatch='/')
    # Negative samples
    plt.bar(indices + bar_width, hist_test_neg, color='blue', width=bar_width, edgecolor='grey', label='Negative', hatch='\\')
    
    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} CN Distribution', fontsize=24)
    plt.xlabel('Num of CN', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'pos_neg_{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to pos_neg_{dataset}_{use_heuristic}_test_distribution.pdf')



def plot_shortest_path_dist(A, split_edge, dataset):
    use_heuristic = 'shortest_path'
    G = from_scipy_sparse_array(A)
    pos_train_pred = shortest_path(G, split_edge['train']['edge'])

    neg_edge_index = negative_sampling(
    data.edge_index, num_nodes=num_nodes,
    num_neg_samples=pos_train_pred.size(0))
    neg_train_pred = shortest_path(G, neg_edge_index.t())

    pos_valid_pred = shortest_path(G, split_edge['valid']['edge'])
    neg_valid_pred = shortest_path(G, split_edge['valid']['edge_neg'])
   
    pos_test_pred = shortest_path(G, split_edge['test']['edge'])
    neg_test_pred = shortest_path(G, split_edge['test']['edge_neg'])

    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    colors = sns.color_palette("husl", 3)  # 'husl' is one of many color palettes available

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist_test, bin_edges = np.histogram(test_pred, bins=bin_edges)
    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    hist_train, bin_edges = np.histogram(train_pred, bins=bin_edges)
    valid_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    hist_valid, bin_edges = np.histogram(valid_pred, bins=bin_edges)
    colors = sns.color_palette("husl", 3)

    bar_width = 0.25
    indices = np.arange(len(hist_train))
    hist_train = hist_train / hist_train.max()
    hist_valid = hist_valid / hist_valid.max()
    hist_test = hist_test / hist_test.max()
    
    plt.figure(figsize=(8, 6))
    plt.bar(indices, hist_train, bar_width, color=colors[0], alpha=0.7, label='Train')
    plt.bar(indices + bar_width, hist_valid, bar_width, color=colors[1], alpha=0.7, label='Valid')
    plt.bar(indices + 2 * bar_width, hist_test, bar_width, color=colors[2], alpha=0.7, label='Test')

    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} SP Distribution', fontsize=24)
    plt.xlabel('Num of SP', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to {dataset}_{use_heuristic}_distribution.pdf')
    
    plt.figure(figsize=(8, 6))
    hist_test_pos, bin_edges = np.histogram(pos_test_pred, bins=bin_edges)
    hist_test_neg, bin_edges = np.histogram(neg_test_pred, bins=bin_edges)
    plt.bar(indices, hist_test_pos, color='green', width=bar_width, edgecolor='grey', label='Positive', hatch='/')
    # Negative samples
    plt.bar(indices + bar_width, hist_test_neg, color='blue', width=bar_width, edgecolor='grey', label='Negative', hatch='\\')
    
    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} Shortest Path Distribution', fontsize=24)
    plt.xlabel('Num of SP', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'pos_neg_{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to pos_neg_{dataset}_{use_heuristic}_test_distribution.pdf')
    return 
    
def plot_spath_citation2_dist(A, split_edge, dataset):
    use_heuristic = 'shortest_path'
    len = 100
    G = from_scipy_sparse_array(A)

    # create edge_index from source and target node
    pos_edge_index  = torch.stack([split_edge['train']['source_node'], split_edge['train']['target_node']])
    edge_index  = {'source_node': split_edge['train']['source_node'], 'target_node': split_edge['train']['target_node']}
    pos_train_pred  =  shortest_path_citation2(G, edge_index)
    neg_edge_index = negative_sampling(
    pos_edge_index, num_nodes=num_nodes,
    num_neg_samples=100)
    
    neg_train_index = {
        'source_node': neg_edge_index[0],
        'target_node': neg_edge_index[1],
    }

    neg_train_pred = shortest_path_citation2(G, neg_train_index)
    
    pos_valid_pred = shortest_path_citation2(G, split_edge['valid'])
    # 1000 copies last 300 hours, please optimize 
    source = split_edge['valid']['source_node'].view(-1, 1).repeat(1, 1000).view(-1)
    target_neg = split_edge['valid']['target_node_neg'].view(-1)

    valid_neg_edge = {'source_node': source[:86596], 'target_node': target_neg[:86596]}
    neg_valid_pred = shortest_path_citation2(G, valid_neg_edge)

    pos_test_pred = shortest_path_citation2(G, split_edge['test'])
    
    source = split_edge['test']['source_node'].view(-1, 1).repeat(1, 1000).view(-1)
    target_neg = split_edge['test']['target_node_neg'].view(-1)

    test_neg_edge = {'source_node': source, 'target_node': target_neg}
    neg_test_pred = shortest_path_citation2(G, test_neg_edge)
    
    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    colors = sns.color_palette("husl", 3)  # 'husl' is one of many color palettes available

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist_test, bin_edges = np.histogram(test_pred, bins=bin_edges)
    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    hist_train, bin_edges = np.histogram(train_pred, bins=bin_edges)
    valid_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    hist_valid, bin_edges = np.histogram(valid_pred, bins=bin_edges)
    colors = sns.color_palette("husl", 3)

    bar_width = 0.25
    indices = np.arange(hist_train.size)
    hist_train = hist_train / hist_train.max()
    hist_valid = hist_valid / hist_valid.max()
    hist_test = hist_test / hist_test.max()
    
    plt.figure(figsize=(8, 6))
    plt.bar(indices, hist_train, bar_width, color=colors[0], alpha=0.7, label='Train')
    plt.bar(indices + bar_width, hist_valid, bar_width, color=colors[1], alpha=0.7, label='Valid')
    plt.bar(indices + 2 * bar_width, hist_test, bar_width, color=colors[2], alpha=0.7, label='Test')

    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} Shortest Path Distribution', fontsize=24)
    plt.xlabel('Shortest Path', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to {dataset}_{use_heuristic}_distribution.pdf')
    
    plt.figure(figsize=(8, 6))
    hist_test_pos, bin_edges = np.histogram(pos_test_pred, bins=bin_edges)
    hist_test_neg, bin_edges = np.histogram(neg_test_pred, bins=bin_edges)
    plt.bar(indices, hist_test_pos, color='green', width=bar_width, edgecolor='grey', label='Positive', hatch='/')
    # Negative samples
    plt.bar(indices + bar_width, hist_test_neg, color='blue', width=bar_width, edgecolor='grey', label='Negative', hatch='\\')
    
    plt.xticks(indices + bar_width, ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{dataset} Shortest Path Distribution', fontsize=24)
    plt.xlabel('Num of SP', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'pos_neg_{dataset}_{use_heuristic}_distribution.pdf')
    print(f'save to pos_neg_{dataset}_{use_heuristic}_test_distribution.pdf')
    


if __name__ == '__main__':
    
    fast_split = True
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                '#bcbd22', '#17becf', '#ff9896']
    # data_list = [
    #             'ogbl-citation2',
    #             'Cora', 'CiteSeer', 'PubMed', 'custom-cora', 'custom-arxiv_2023', 
    #             'custom-pubmed', 'custom-ogbn-arxiv', 
    #             'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
    #             'ogbl-vessel', 
    #         # 'Cora', 
    # ]


    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--data', dest='data', nargs='+', choices=['Cora', 'CiteSeer', 'PubMed', 'custom-cora', 'custom-arxiv_2023', 
                                                            'custom-pubmed', 'custom-ogbn-arxiv',
                                                            'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                                                            'ogbl-vessel'],
                                    #default='ogbl-ppa',
        help='List of datasets')

    args = parser.parse_args()


    plot_degree = False
    plot_clustering = False
    plot_CN_dist_link = False
    plot_shortest_path_link = True
    
    data_list = args.data
    plt.figure(figsize=(8, 6))
    fast_split = True

    for dataset, color in zip(data_list, color_list):
        print(dataset)
        try:
            data, splits = load_graph(dataset)
        except ValueError as e:
            print(e)
            continue

        print(f"Dataset: {dataset}")
        adj = to_scipy_sparse_matrix(data.edge_index)
        train_edge_index = data.edge_index
        edge_weight = torch.ones(train_edge_index.size(1))
        num_nodes = data.num_nodes

        A = ssp.csr_matrix((edge_weight.view(-1), (train_edge_index[0], train_edge_index[1])), shape=(num_nodes, num_nodes)) 

        if plot_degree:
            G = from_scipy_sparse_array(adj)
            plot_degree_distribution(G, color, dataset)
        elif plot_clustering:
            G = from_scipy_sparse_array(adj)
            plot_clustering_specturm(G, color, dataset)

        elif plot_CN_dist_link:
            if dataset != 'ogbl-citation2':
                plot_CN_dist(A, splits, dataset)
            else:
                plot_CN_citation2_dist(A, splits, dataset)
        elif plot_shortest_path_link:
            if dataset != 'ogbl-citation2':
                plot_shortest_path_dist(A, splits, dataset)
            else:
                plot_spath_citation2_dist(A, splits, dataset)
            

    # plt.legend(loc='upper right')
    # plt.xlabel('degree d')
    # plt.ylabel('fraction of vertices with degree ≥ d')
    # plt.title(f'Degree Distribution for {dataset}')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.savefig('gather_degree_distribution.pdf')



def plot_CN_pos_neg_dist():
    # Example data: replace these with your actual CN distributions
    # These are hypothetical counts for the CN ranges for both positive and negative samples
    pos_counts = [45.5, 11.2, 20.1, 23.3]
    neg_counts = [99.94, 0.04, 0.02, 0.01]
    cn_ranges = ['[0, 1)', '[1, 3)', '[3, 10)', '[10, ∞)']

    # Calculate the percentage of samples
    pos_percent = [count / sum(pos_counts) * 100 for count in pos_counts]
    neg_percent = [count / sum(neg_counts) * 100 for count in neg_counts]

    # Set up the bar width and positions
    bar_width = 0.4
    r1 = np.arange(len(pos_counts))
    r2 = [x + bar_width for x in r1]

    # Plotting
    plt.figure(figsize=(10, 7))

    # Positive samples
    plt.bar(r1, pos_percent, color='green', width=bar_width, edgecolor='grey', label='Positive', hatch='/')

    # Negative samples
    plt.bar(r2, neg_percent, color='blue', width=bar_width, edgecolor='grey', label='Negative', hatch='\\')

    # Adding percentages on top of bars
    for i in range(len(r1)):
        plt.text(r1[i] - 0.1, pos_percent[i] + 1, f'{pos_percent[i]:.2f}%', color='black', fontweight='bold')
        plt.text(r2[i] - 0.1, neg_percent[i] + 1, f'{neg_percent[i]:.2f}%', color='black', fontweight='bold')

    # Customizing the plot
    plt.xlabel('Number of CNs', fontweight='bold', fontsize=15)
    plt.ylabel('% of Samples', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width/2 for r in range(len(r1))], cn_ranges, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Common Neighbor Distribution', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()
