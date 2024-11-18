"""
utils for getting the largest connected component of a graph
"""
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import torch
from torch_sparse.tensor import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from scipy.sparse import coo_matrix
from tqdm import tqdm
from typing import List
import torch_geometric.utils as pyg_utils
import networkx as nx
import time

def get_Data(data: Data):
  if type(data) is InMemoryDataset:
    return data._data
  elif type(data) is Data:
    return data
  elif type(data) is PygNodePropPredDataset or type(data) is PygLinkPropPredDataset:
    return data._data
  else:
    return data[0]


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  data = get_Data(dataset)

  remaining_nodes = set(range(data.x.shape[0]))

  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)

    remaining_nodes = remaining_nodes.difference(comp)

  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def get_row_col(edge_index: SparseTensor) -> list:
  if type(edge_index) is torch.Tensor:
    row, col = edge_index.numpy()
  elif type(edge_index) is SparseTensor:
    row, col = edge_index.to_torch_sparse_coo_tensor().coalesce().indices().numpy()
  return row, col


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  # BFS
  """this function detect the llc of a undirected graph with symmetric adjacency matrix

  Args:
      dataset (InMemoryDataset): modified graph from ogb or planetoid
      start (int, optional): start node index. Defaults to 0.
      dataset.data.edge_index is a tensor of shape [2, num_edges], default type is numpy.ndarray

  Returns:
      set: return a set of the node set of local connected component of the graph
  """
  data = get_Data(dataset)
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = get_row_col(data.edge_index)

  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_comp_data(adjacencyList: List[List[int]], start: int = 0) -> set:
  num_nodes = len(adjacencyList)
  visited = [False] * num_nodes
  queued_nodes = [start]
  visited_nodes = set()

  while len(queued_nodes) != 0:
    current_node = queued_nodes.pop()
    if visited[current_node]:
      continue

    visited[current_node] = True
    visited_nodes.add(current_node)
    neighbors = adjacencyList[current_node] # this call costed O(n) time before. Now it only needs O(#neighbors)
    queued_nodes.extend(neighbors)
  return visited_nodes

def construct_sparse_adj(edge_index) -> coo_matrix:
    """
    Construct a sparse adjacency matrix from an edge index.

    Parameters:
    - edge_index: np.array or tuple, edge index
    """
    # Resource: https://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern

    if type(edge_index) == tuple:
        edge_index = np.concatenate([[edge_index[0].numpy()],
                                     [edge_index[1].numpy()]], axis=0)
    elif type(edge_index) != np.ndarray:
        edge_index.numpy()

    if edge_index.shape[0] > edge_index.shape[1]:
        edge_index = edge_index.T

    rows, cols = edge_index[0, :], edge_index[1, :]
    vals = np.ones_like(rows)
    shape = (edge_index.max() + 1, edge_index.max() + 1)
    m = coo_matrix((vals, (rows, cols)), shape=shape)
    return m

def use_lcc(dataset: InMemoryDataset) -> InMemoryDataset:
    # lcc = get_largest_connected_component(dataset)
    m = construct_sparse_adj(dataset.edge_index.numpy())

    start = time.time()
    try:  # for undirected graph
      G = nx.from_scipy_sparse_matrix(m)
    except:
      G = nx.from_scipy_sparse_array(m)
    print('create graph:', time.time() - start)

    for i, c in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
      if i == 0:
        print([len(c)])
        G = G.subgraph(c)
      if i > 0 and i < 5:
        print([len(c)])
      else:
        break

    lcc_index = list(max(nx.connected_components(G), key=len))
    data = get_Data(dataset)

    if data.x is not None:
      x_new = data.x[lcc_index]
    else:
      x_new = None

    row, col = get_row_col(data.edge_index)

    lcc_set = set(lcc_index)
    mask = np.array([(i in lcc_set and j in lcc_set) for i, j in zip(row, col)])
    filtered_edges = np.column_stack((row[mask], col[mask]))
    node_mapper = get_node_mapper(lcc_index)
    edges = remap_edges(filtered_edges, node_mapper)

    new_data = Data(
        x = x_new,
        edge_index = torch.LongTensor(edges),
        # y=y_new,
        num_nodes = torch.LongTensor(edges).max().tolist()+1,
        node_attrs = x_new,
        edge_attrs = None,
        graph_attrs = None
    )

    return new_data, lcc_index, G
  

def find_scc_direc(data) -> List:
    # Convert PyTorch Geometric graph to NetworkX graph
    G = pyg_utils.to_networkx(data, to_undirected=False)
    
    # Compute strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    
    # Find the largest strongly connected component
    largest_scc = max(sccs, key=len)
    
    return list(largest_scc)
  
  
def use_lcc_direc(data, lcc):
    # Create a mask for the nodes in the largest SCC
    lcc = list(lcc)
    x_new = data.x[lcc]
    
    row, col = get_row_col(data.edge_index)
    edges = [[i, j] for i, j in tqdm(zip(row, col)) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    subgraph = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        # y=y_new,
        num_nodes=x_new.size()[0],
        # train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        # test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        # val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        node_attrs=x_new, 
        edge_attrs = None, 
        graph_attrs = None
    )
    return subgraph



# if __name__ == '__main__':
#   import numpy as np
#   from torch_geometric.data import Data, InMemoryDataset
#   import torch 
#   import matspy as spy

#   from torch_geometric.datasets import Planetoid, KarateClub
#   from torch_geometric.data import Data, InMemoryDataset
#   from ogb.linkproppred import PygLinkPropPredDataset
#   import numpy as np
#   import torch
    # params
    # path = '.'
    # for name in ['cora', 'pubmed', 'ogbn-arxiv', 'ogbn-products', 'arxiv_2023']:
        
    #     if name in ['cora', 'pubmed', 'citeseer', 'ogbn-arxiv', 'ogbn-products', 'arxiv_2023']:
    #         use_lcc_flag = True
            
    #         # planetoid_data = Planetoid(path, name) 
    #         data, num_class, text = load_data(name, use_dgl=False, use_text=False, use_gpt=False, seed=0)
            
    #         print(f" original num of nodes: {data.num_nodes}")
            
    #         if name.startswith('ogb'):
    #             edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
    #         else:
    #             edge_index = data.edge_index.numpy()
                
    #         m = construct_sparse_adj(edge_index)
    #         fig, ax = spy.spy_to_mpl(m)
    #         fig.savefig(f"plots/{name}/{name}_ori_data_index_spy.png", bbox_inches='tight')
            
            
    #         if use_lcc_flag:
    #             data_lcc = use_lcc(data)
    #         print(data_lcc.num_nodes)


    #         m = construct_sparse_adj(data_lcc.edge_index.numpy())
    #         fig, ax = spy.spy_to_mpl(m)
    #         fig.savefig(f"plots/{name}/{name}_lcc_data_index_spy.png", bbox_inches='tight')
        