import torch
from torch.utils.data import Dataset as TorchDataset

# Create torch dataset
class LinkPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, edge_index, labels=None):

        self.encodings = encodings
        self.edge_index = edge_index.cpu()
        self.labels = labels.long().cpu()

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
