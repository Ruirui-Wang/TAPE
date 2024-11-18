import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.transforms import RandomLinkSplit

from graph_stats import (construct_sparse_adj, 
                         from_scipy_sparse_array,
                         plot_all_cc_dist, 
                         graph_metrics_nx )
from time import time 
import pandas as pd 
import networkx as nx 
import time 
from typing import Dict, Tuple, List, Union
from graph_stats import (graph_metrics_nx, 
                         _counts, 
                         _degrees, 
                         _gini_coefficient, 
                         _avg_degree, 
                         _avg_degree2, 
                         _degree_heterogeneity, 
                         _avg_shortest_path, 
                         _diameter, 
                         _power_law_estimate,
                         _largest_connected_component_size)
import numpy as np
from graph_stats import feat_homophily
from data_utils.load_data_lp import  get_edge_split


# random split dataset
def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data = dataset[0]
    data.num_nodes = data.x.shape[0]
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

def loaddataset(name: str, use_valedges_as_input: bool, load=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    if name == "ppa":
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi":
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge


def graph_metrics_nx(graph: nx.Graph, name: str, use_lcc: bool) -> Dict[str, float]:
    """Computes graph metrics on a networkx graph object.

    Arguments:
        graph: networkx graph.
    Returns:
        dict from metric names to metric values.
    """
    result = {'name': f"{name}_{use_lcc}"}
    result.update(_counts(graph))
    degrees = _degrees(graph)
    result['degree_gini'] = _gini_coefficient(degrees)
    
    avg_degree_G, avg_degree_dict = _avg_degree(G)
    avg_degree_G2 = _avg_degree2(graph, avg_degree_dict)
    result['avg_deg'] = avg_degree_G
    result['avg_deg2'] = avg_degree_G2

    result['deg_heterogeneity'] = _degree_heterogeneity(graph)
    result['avg_shortest_path'] = _avg_shortest_path(graph, name, 1000) if nx.is_connected(G) else np.inf
    
    result['approximate_diameter'] = np.inf 
    result['num_triangles'] = float(
    np.sum(list(nx.triangles(graph).values())) / 3.0)
        
    if graph.number_of_nodes() == 0:  # avoid np.mean of empty slice
        result['avg_degree'] = 0.0
        return result
    
    result['avg_degree'] = float(np.mean(degrees))
    core_numbers = np.array(list(nx.core_number(graph).values()))
    result['coreness_eq_1'] = float(np.mean(core_numbers == 1))
    result['coreness_geq_2'] = float(np.mean(core_numbers >= 2))
    result['coreness_geq_5'] = float(np.mean(core_numbers >= 5))
    result['coreness_geq_10'] = float(np.mean(core_numbers >= 10))
    result['coreness_gini'] = float(_gini_coefficient(core_numbers))
    result['avg_cc'] = float(np.mean(list(nx.clustering(graph).values())))
    result['transitivity'] = float(nx.transitivity(graph))
    
    result['cc_size'] = float(_largest_connected_component_size(graph))
    result['power_law_estimate'] = _power_law_estimate(degrees)
    return result

if __name__ == "__main__":
    gc = [] #"ppa", "collab", "citation2", "vessel"
    for name in ["citation2"]:
        data,  split_edge = loaddataset(name, False)
        start_time = time.time()
        m = construct_sparse_adj(data.edge_index.numpy())
        G = from_scipy_sparse_array(m)
        print(f"Time taken to create graph: {time.time() - start_time} s")
        
        if  False:
            plot_all_cc_dist(G, 'haha')
        
        if True:
            gc.append(graph_metrics_nx(G, name, False))
            print(gc)
            
            gc = pd.DataFrame(gc)
            gc.to_csv(f'{name}_all_graph_metric_False.csv', index=False)
            
    """
    gc = [] 
    results = [] 
    for name in ["ppa", "ddi", "collab", "citation2", "vessel"]:
        data,  split_edge = loaddataset(name, False)
        data.x.float()
        results.append(feat_homophily(split_edge, data.x, name))
            
    print(results)
    df = pd.DataFrame(results)

    # Save DataFrame to CSV
    csv_file_path = 'ogb_feat_results.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"Results saved to {csv_file_path}")
    """