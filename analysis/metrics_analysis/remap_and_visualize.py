import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data

from data_utils.load import load_data_lp




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--name', type=str, required=False, default='cora')
    parser.add_argument('--undirected', type=bool, required=False, default=True)
    parser.add_argument('--include_negatives', type=bool, required=False, default=True)
    parser.add_argument('--val_pct', type=float, required=False, default=0.15)
    parser.add_argument('--test_pct', type=float, required=False, default=0.05)
    parser.add_argument('--split_labels', type=bool, required=False, default=True)
    parser.add_argument('--device', type=str, required=False, default='cpu')
    return parser.parse_args()

def remap_node_indices(data, splits):
    edge_index = data.edge_index
    degrees = torch.bincount(edge_index.view(-1))
    sorted_indices = torch.argsort(degrees, descending=True)
    index_map = torch.empty_like(sorted_indices)
    index_map[sorted_indices] = torch.arange(len(sorted_indices))
    new_edge_index = index_map[edge_index]
    if data.x is not None:
        new_x = data.x[sorted_indices]
    else:
        new_x = None
    data.edge_index = new_edge_index
    data.x = new_x
    for set in ['train', 'valid', 'test']:
        splits[set].x = new_x
        edge_index_device = splits[set].edge_index.device
        index_map = index_map.to(edge_index_device)
        splits[set].edge_index = index_map[splits[set].edge_index]
        splits[set].pos_edge_label_index = index_map[splits[set].pos_edge_label_index]
        splits[set].neg_edge_label_index = index_map[splits[set].neg_edge_label_index]
    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        num_nodes=data.num_nodes
    )
    return new_data, splits


def visualize_adjacency_matrix(name, num_nodes, pos_edge_index=None, neg_edge_index=None):
    if num_nodes > 100000:
        fig, ax = plt.subplots(figsize=(40, 40))
    else:
        fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim(-0.5, num_nodes - 0.5)
    ax.set_ylim(-0.5, num_nodes - 0.5)

    if pos_edge_index is not None:
        pos_edges = pos_edge_index.t().numpy()
        ax.scatter(pos_edges[:, 1], pos_edges[:, 0], color='green', label='Positive Edges', s=10)

    if neg_edge_index is not None:
        neg_edges = neg_edge_index.t().numpy()
        ax.scatter(neg_edges[:, 1], neg_edges[:, 0], color='red', label='Negative Edges', s=10)

    plt.title(f'Adjacency Matrix - {name}')
    plt.legend()
    plt.savefig('./visualize/'+ f'Adjacency Matrix - {name}')
    plt.close()

def visualize_weighted_adjacency_matrix(name, num_nodes, pos_edge_index, neg_edge_index, pos_weight, neg_weight):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if num_nodes > 100000:
        fig, ax = plt.subplots(figsize=(40, 40))
    else:
        fig, ax = plt.subplots(figsize=(20, 20))

    ax.set_xlim(-0.5, num_nodes - 0.5)
    ax.set_ylim(-0.5, num_nodes - 0.5)
    ax.set_aspect('equal')  # Ensure the plot is square
    pos_edges = pos_edge_index.t().cpu().numpy()
    neg_edges = neg_edge_index.t().cpu().numpy()
    pos_weight = pos_weight.cpu().detach().numpy()
    neg_weight = neg_weight.cpu().detach().numpy()

    pos_norm = (pos_weight - pos_weight.min()) / (pos_weight.max() - pos_weight.min())
    neg_norm = 1 - (neg_weight - neg_weight.min()) / (
                neg_weight.max() - neg_weight.min())  # Reverse normalization for negative weights

    # Use different colormaps for positive and negative edges
    cmap_pos = plt.cm.Blues  # Color map for positive edges
    cmap_neg = plt.cm.Reds  # Color map for negative edges
    pos_colors = cmap_pos(pos_norm)
    neg_colors = cmap_neg(neg_norm)

    # Plot positive edges with color mapping
    scatter_pos = ax.scatter(pos_edges[:, 1], pos_edges[:, 0], color=pos_colors, label='Positive Edges', s=100)

    # Plot negative edges with color mapping
    scatter_neg = ax.scatter(neg_edges[:, 1], neg_edges[:, 0], color=neg_colors, label='Negative Edges', s=100)

    plt.title(f'Adjacency Matrix - {name}')
    legend = plt.legend(loc='upper right')
    plt.gca().add_artist(legend)  # Add legend without it overlapping the colorbars

    # Create a new axis for the colorbars to ensure they have the same length
    cbar_ax_pos = fig.add_axes([0.91, 0.55, 0.02, 0.35])  # Adjust the position as needed
    cbar_ax_neg = fig.add_axes([0.91, 0.1, 0.02, 0.35])  # Adjust the position as needed

    # Normalize both weights to the 0-1 range
    norm = Normalize(vmin=0, vmax=1)

    # Create colorbars for both scatter plots
    sm_pos = ScalarMappable(cmap=cmap_pos, norm=norm)
    sm_pos.set_array([])
    cbar_pos = plt.colorbar(sm_pos, cax=cbar_ax_pos)
    cbar_pos.set_label('Positive Edge Weight')

    sm_neg = ScalarMappable(cmap=cmap_neg, norm=norm)
    sm_neg.set_array([])
    cbar_neg = plt.colorbar(sm_neg, cax=cbar_ax_neg)
    cbar_neg.set_label('Negative Edge Weight')

    plt.savefig('./visualize/' + f'Adjacency Matrix - {name}')
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    args.split_index = [0.8, 0.15, 0.05]
    if not os.path.exists('../core/data_utils/visualize'):
        os.makedirs('../core/data_utils/visualize')
    for dataset in ['pwc_small', 'cora', 'pubmed', 'arxiv_2023', 'ogbn-arxiv', 'citationv8', 'pwc_medium', 'pwc_large']:
        args.name = dataset
        splits, text, data = load_data_lp[dataset](args)
        data, splits = remap_node_indices(data, splits)
        visualize_adjacency_matrix(f'{dataset}_all', data.num_nodes, data.edge_index)
        visualize_adjacency_matrix(f'{dataset}_train', splits['train'].num_nodes,
                                   splits['train'].pos_edge_label_index, splits['train'].neg_edge_label_index)
        visualize_adjacency_matrix(f'{dataset}_valid', splits['valid'].num_nodes,
                                   splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index)
        visualize_adjacency_matrix(f'{dataset}_test',splits['test'].num_nodes,
                                   splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index)
        print(f"Data remapping completed for {dataset}.")

