import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def visualize_weighted_adjacency_matrix(name, num_nodes, pos_edge_index, neg_edge_index, pos_weight, neg_weight):
    # Determine plot size based on the number of nodes
    fig_size = (40, 40) if num_nodes > 100000 else (20, 20)
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_xlim(-0.5, num_nodes - 0.5)
    ax.set_ylim(-0.5, num_nodes - 0.5)
    ax.set_aspect('equal')  # Ensure the plot is square

    # Extract edges and weights
    pos_edges = pos_edge_index.t().cpu().numpy()
    neg_edges = neg_edge_index.t().cpu().numpy()
    pos_weight = pos_weight.cpu().detach().numpy()
    neg_weight = neg_weight.cpu().detach().numpy()

    # Normalize weights
    pos_norm = (pos_weight - pos_weight.min()) / (pos_weight.max() - pos_weight.min())
    neg_norm = 1 - (neg_weight - neg_weight.min()) / (neg_weight.max() - neg_weight.min())

    # Use different colormaps for positive and negative edges
    cmap_pos = plt.cm.Blues  # Color map for positive edges
    cmap_neg = plt.cm.Reds  # Color map for negative edges
    pos_colors = cmap_pos(pos_norm)
    neg_colors = cmap_neg(neg_norm)

    # Compute node degrees for sizing the scatter points
    pos_degrees = Counter(pos_edges.flatten())
    neg_degrees = Counter(neg_edges.flatten())

    # Size by degree (you can adjust the scaling factor as needed)
    def get_sizes(edges):
        degrees = Counter(edges.flatten())
        sizes = np.array([degrees[node] for node in edges[:, 0]])
        return sizes * 10  # Adjust the multiplier for desired point size

    pos_sizes = get_sizes(pos_edges)
    neg_sizes = get_sizes(neg_edges)

    # Plot positive edges with color mapping and sizes
    scatter_pos = ax.scatter(pos_edges[:, 1], pos_edges[:, 0], color=pos_colors, label='Positive Edges', s=pos_sizes)

    # Plot negative edges with color mapping and sizes
    scatter_neg = ax.scatter(neg_edges[:, 1], neg_edges[:, 0], color=neg_colors, label='Negative Edges', s=neg_sizes)

    plt.title(f'Adjacency Matrix - {name}')
    legend = plt.legend(loc='upper right')
    plt.gca().add_artist(legend)  # Add legend without it overlapping the colorbars

    # Create new axes for the colorbars to ensure they have the same length
    cbar_ax_pos = fig.add_axes([0.91, 0.55, 0.02, 0.35])  # Adjust the position as needed
    cbar_ax_neg = fig.add_axes([0.91, 0.1, 0.02, 0.35])   # Adjust the position as needed

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

    plt.savefig(f'./visualize/Adjacency_Matrix_{name}.png')
    plt.close()