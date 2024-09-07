import torch
from torch_geometric.data import Data
from torch_geometric.loader.cluster import ClusterData
import dgl 
import torch as th
from dgl.data import RedditDataset, YelpDataset
from dgl.distributed import partition_graph
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.utils import to_torch_coo_tensor
from ogb.nodeproppred import NodePropPredDataset
import networkx as nx 
from scipy.sparse import csc_array
import numpy as np
import matspy as spy
from data_utils.load import load_data_nc

def load_ogb_dataset(name, data_path):
    if name.startswith('ogbn'):
        dataset = DglNodePropPredDataset(name=name, root=data_path)
        split_idx = dataset.get_idx_split()
        g, label = dataset[0]
        n_node = g.num_nodes()
        node_data = g.ndata
        node_data['label'] = label.view(-1).long()
        node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
        node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
        node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
        node_data['train_mask'][split_idx["train"]] = True
        node_data['val_mask'][split_idx["valid"]] = True
        node_data['test_mask'][split_idx["test"]] = True
    else:
        raise NotImplementedError
    return g


def ogb_to_dgl(data):
    
    # Extract node features, edge index, and labels
    x = data[0]['node_feat'] # Node features
    edge_index = data[0]['edge_index'] # Edge index
    y = data[1] # Node labels

    # Convert to DGLGraph
    g = dgl.DGLGraph()
    g.add_nodes(x.shape[0])  # Number of nodes
    g.add_edges(edge_index[0], edge_index[1])  # Edges (from PyG tuple format

    # Set node features and labels
    g.ndata['feat'] = torch.FloatTensor(x)
    g.ndata['label'] = torch.tensor(y[:, 0], dtype=torch.long)
    
    return g

def tag_to_dgl(data):
    # transform pytorch geometric data to dgl graph
    if type(data) == dgl.DGLGraph:
        return data
    
    print(data)
    # transform to dgl graph
    g = dgl.DGLGraph()
    g.add_nodes(data.num_nodes)  # Number of nodes
    g.add_edges(data.edge_index[0], data.edge_index[1])  # Edges (from PyG tuple format
    g.ndata['feat'] = data.x
    g.ndata['label'] = data.y
    g.ndata['train_mask'] = data.train_mask
    g.ndata['val_mask'] = data.val_mask
    g.ndata['test_mask'] = data.test_mask
    return g

def egde_index_to_dense(edge_index, num_nodes):
    # transform sparse form into dense form
    row, col = edge_index
    data = np.ones(row.shape[0])
    return csc_array((data, (row, col)), shape=(num_nodes, num_nodes))

def compare_remap(data) -> None:
    # compare the adjacency matrix of the original graph and the remapped graph
    # plot original adjacency matrix
    g = tag_to_dgl(data)

    num_nodes = g.num_nodes()
    adjacency = egde_index_to_dense(g.edges(), num_nodes)
    fig, ax = spy.spy_to_mpl(adjacency)
    fig.savefig(f"plots/{name}/{name}_data_edges_spy.png", bbox_inches='tight')
    
    # partition the graph and node mapping
    node_map, edge_map = partition_graph(g, 'test', num_clusters, out_path='output/',  part_method='metis',
                                balance_edges=False, return_mapping=True)
    
    # remap 
    remapped_node_features = g.ndata['feat'][node_map]
    
    # Create a new graph with remapped features
    remapped_g = dgl.DGLGraph()
    remapped_g.add_nodes(num_nodes)  # Number of nodes
    remapped_g.add_edges(node_map[g.edges()[0].numpy()], node_map[g.edges()[1].numpy()])
    remapped_g.ndata['features'] = remapped_node_features

    adj_sparse = remapped_g.edges()
    
    # replot remapped adjacency matrix55
    remapped_adjacency = egde_index_to_dense(adj_sparse, num_nodes)
    fig, ax = spy.spy_to_mpl(remapped_adjacency)
    fig.savefig(f"plots/{name}/remapped{name}_data_edges_spy.png", bbox_inches='tight')
    
    print(np.array_equal(adjacency.todense(), remapped_adjacency.todense()))
    return remapped_adjacency

def metis_partation(data) -> None:
    # compare the adjacency matrix of the original graph and the remapped graph
    # plot original adjacency matrix
    g = tag_to_dgl(data)

    num_nodes = g.num_nodes()
    adjacency = egde_index_to_dense(g.edges(), num_nodes)
    fig, ax = spy.spy_to_mpl(adjacency)
    fig.savefig(f"plots/{name}/{name}_data_edges_spy.png", bbox_inches='tight')
    
    # partition the graph and node mapping
    node_map, _ = partition_graph(g, 'test', num_clusters, out_path='output/',  part_method='metis',
                                balance_edges=False, return_mapping=True)
    
    # remap 
    remapped_adjacency = node_map[g.edges()[0].numpy()], node_map[g.edges()[1].numpy()]
    
    # replot remapped adjacency matrix55
    remapped_adjacency = egde_index_to_dense(remapped_adjacency , num_nodes)
    fig, ax = spy.spy_to_mpl(remapped_adjacency)
    fig.savefig(f"plots/{name}/remapped{name}_data_edges_spy.png", bbox_inches='tight')
    
    print(np.array_equal(adjacency.todense(), remapped_adjacency.todense()))
    return remapped_adjacency


if __name__ == '__main__':
    # params
    num_clusters = 4
    # 'arxiv_2023', 'ogbn-arxiv', 'ogbn-products', 
    for name in ['cora', 'pubmed']:
        # data
        if name in ['cora', 'citeseer', 'arxiv_2023', 'pubmed']:
            data, num_class, text = load_data_nc[name]()
            
        elif name in ['ogbn-arxiv', 'ogbn-products']:
            data = load_ogb_dataset(name, 'dataset')
        
        # compare_remap(data)
        metis_partation(data)
        
    exit(-1)
    
    
g, node_feats, edge_feats, gpb, graph_name, _, _ = dgl.distributed.load_partition(
                                'output/test.json', 1)

print(g)

subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition('output/test.json', 1)
node_type = node_type[0]
node_feat[dgl.NID] = subg.ndata[dgl.NID]
if 'part_id' in subg.ndata:
    node_feat['part_id'] = subg.ndata['part_id']
node_feat['inner_node'] = subg.ndata['inner_node'].bool()
node_feat['label'] = node_feat[node_type + '/label']
node_feat['feat'] = node_feat[node_type + '/feat']

node_feat.pop(node_type + '/label')
node_feat.pop(node_type + '/feat')

