
import dgl
import torch
from torch.utils.data import Dataset as TorchDataset

# convert PyG dataset to DGL dataset


class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            g = dgl.to_bidirected(g)
            print(
                f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
        if data.x is not None:
            g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask


# Create torch dataset
class LinkPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, edge_index, labels=None):

        self.encodings = encodings
        self.edge_index = edge_index
        self.labels = labels

    def __getitem__(self, idx):
        node1_idx, node2_idx = self.edge_index[:, idx]

        node1_features = {key: torch.tensor(val[node1_idx]) for key, val in self.encodings.items()}
        node2_features = {key: torch.tensor(val[node2_idx]) for key, val in self.encodings.items()}

        item = {key: torch.stack([node1_features[key], node2_features[key]]) for key in node1_features}
        item['node_id'] = self.edge_index[:, idx]
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return self.edge_index.shape[1]
