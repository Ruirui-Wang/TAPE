import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch

from sklearn.metrics import (roc_auc_score, accuracy_score, average_precision_score)
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import torch
import scipy.sparse as ssp
import pandas as pd
from IPython.display import display, Markdown

from core.heuristic.lsf import CN, AA, RA, InverseRA
from core.heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close, SymPPR
from core.data_utils.load import load_data_lp
from core.data_utils.lcc import use_lcc
from core.graphgps.utility.utils import init_cfg_test


def plot_pessimistic_rank(y_pred_pos, y_pred_neg):
    plt.figure(figsize=(12, 8))
    # Plot distributions of probabilities
    plt.subplot(1, 2, 1)
    plt.imshow((y_pred_neg >= y_pred_pos))
    plt.xlabel('Sample Index')
    plt.ylabel('Neg >= Pos')
    plt.title('Distribution of Optimistic Rank')

    plt.subplot(1, 2, 2)
    plt.imshow((y_pred_neg > y_pred_pos))
    plt.xlabel('Sample Index')
    plt.ylabel('Pos > Pos')
    plt.title('Distribution of Pessimistic Rank')
    plt.savefig('optimistic_rank.png')


def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    results = {}
    valid_auc = round(valid_auc, 4)
    results['AUC'] = valid_auc
    valid_ap = average_precision_score(val_true, val_pred)
    valid_ap = round(valid_ap, 4)
    results['AP'] = valid_ap
    return results


def error_mrr_pos(y_pred_pos, pos_edge_index, y_pred_neg, k_list):
    # add one axis
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    # ~> the positive is ranked last among those with equal score
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    plot_rank_list(ranking_list, 'pos')
    plt.figure(figsize=(10, 6))
    plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')
    plt.title('Histogram of Tensor Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # plt.savefig('pos_2neg_hist.png')
    # Define the error range (x-axis range considered as error)
    norm_interval = [0.2, 1.0]
    error_ranges = [
        [int(i * ranking_list.max().item()) - 1 for i in norm_interval]]  # Example error ranges, adjust as needed

    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')

    for i in range(len(patches)):
        bin_center = 0.5 * (bins[i] + bins[i + 1])
        for error_range in error_ranges:
            if error_range[0] <= bin_center <= error_range[1]:
                patches[i].set_facecolor('r')
                break
            else:
                patches[i].set_facecolor('b')
                # Add labels and title
    plt.title('Histogram of Tensor Distribution with Error Ranges Highlighted, Pos ranked lower than Neg')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    # plt.savefig('pos_2neg_hist.png')

    len_rank = ranking_list.shape[0]
    result = {}
    for k in k_list:
        result[f'mrr_hit{round(k / len_rank, 2)}'] = (ranking_list <= k).sum(0) / len_rank

    ranking_list = ranking_list.cpu().numpy().astype(int)
    # rank_interval = ranking_list.min() + ((ranking_list.max() - ranking_list.min()) * np.array(norm_interval)).astype(int)
    error_mask = np.zeros(ranking_list.shape[0], dtype=bool)
    for errors in error_ranges:
        error_mask = error_mask | ((ranking_list > errors[0]) & (ranking_list < errors[1]))

    pos_edge_index_err = pos_edge_index[error_mask, :]
    pos_rank_err = ranking_list[error_mask]

    return result, pos_edge_index_err, pos_rank_err


def error_mrr_neg(y_pred_neg, neg_edge_index, y_pred_pos, k_list):
    # calculate ranks
    y_pred_neg = y_pred_neg.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg <= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg < y_pred_pos).sum(dim=1)

    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) - 1

    plot_rank_list(ranking_list, 'neg')

    len_rank = ranking_list.shape[0]
    result = {}
    for k in k_list:
        result[f'mrr_hit{round(k / len_rank, 2)}'] = (ranking_list <= k).sum(0) / len_rank

    plt.figure(figsize=(10, 6))
    plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')
    plt.title('Histogram of Tensor Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    norm_interval = [0.0, 0.4]
    error_ranges = [
        [int(i * ranking_list.max().item()) - 1 for i in norm_interval]]  # Example error ranges, adjust as needed
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')
    for i in range(len(patches)):
        bin_center = 0.5 * (bins[i] + bins[i + 1])
        for error_range in error_ranges:
            if error_range[0] <= bin_center <= error_range[1]:
                patches[i].set_facecolor('r')
                break
            else:
                patches[i].set_facecolor('b')
    plt.title('Histogram of Tensor Distribution with Error Ranges Highlighted, Neg ranked lower than Pos')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    ranking_list = ranking_list.cpu().numpy().astype(int)
    error_mask = np.zeros(ranking_list.shape[0], dtype=bool)
    for errors in error_ranges:
        error_mask = error_mask | ((ranking_list > errors[0]) & (ranking_list < errors[1]))

    neg_edge_index_err = neg_edge_index[error_mask, :]
    neg_rank_err = ranking_list[error_mask]

    return result, neg_edge_index_err, neg_rank_err


def get_metric_invariant(
        pos_test_pred: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_test_pred: torch.Tensor,
        neg_edge_index: torch.Tensor,
        k_list: List[float]
) -> Tuple[float, float, Dict[str, float], float, float, float, float]:
    """
    Computes metrics for evaluating link prediction models.

    Parameters:
        pos_test_pred (torch.Tensor): Tensor of positive test predictions.
        pos_edge_index (torch.Tensor): Tensor of positive edge indices.
        neg_test_pred (torch.Tensor): Tensor of negative test predictions.
        neg_edge_index (torch.Tensor): Tensor of negative edge indices.
        k_list (List[float]): List of float values representing thresholds for ranking.

    Returns:
        Tuple[float, float, Dict[str, float], float, float, float, float]:
            - MRR for positive-to-negative predictions
            - MRR for negative-to-positive predictions
            - Dictionary containing AUC and AP scores
            - Errors in positive edge indices
            - Errors in positive ranks
            - Errors in negative edge indices
            - Errors in negative ranks
    """

    # Ensure inputs are Tensors
    pos_test_pred = torch.as_tensor(pos_test_pred)
    pos_edge_index = torch.as_tensor(pos_edge_index)
    neg_test_pred = torch.as_tensor(neg_test_pred)
    neg_edge_index = torch.as_tensor(neg_edge_index)

    # Ensure k_list is sorted and convert k values to integer indices
    k_rank = sorted(int(k * neg_test_pred.size(0)) for k in k_list)

    # Check tensor dimensions
    if pos_test_pred.size(0) != pos_edge_index.size(0):
        raise ValueError("Size mismatch: pos_test_pred and pos_edge_index")
    if neg_test_pred.size(0) != neg_edge_index.size(0):
        raise ValueError("Size mismatch: neg_test_pred and neg_edge_index")

    # Calculate MRR and errors
    mrr_pos2neg, pos_edge_index_err, pos_rank_err = error_mrr_pos(
        pos_test_pred,
        pos_edge_index,
        neg_test_pred.repeat(pos_test_pred.size(0), 1),
        k_rank
    )
    mrr_neg2pos, neg_edge_index_err, neg_rank_err = error_mrr_neg(
        neg_test_pred,
        neg_edge_index,
        pos_test_pred.repeat(neg_test_pred.size(0), 1),
        k_rank
    )

    # Concatenate predictions and true labels
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([
        torch.ones(pos_test_pred.size(0), dtype=torch.int),
        torch.zeros(neg_test_pred.size(0), dtype=torch.int)
    ])

    # Evaluate AUC and AP
    result_auc_test = evaluate_auc(test_pred, test_true)

    # Ensure result_auc_test contains expected keys
    if not all(key in result_auc_test for key in ['AUC', 'AP']):
        raise KeyError("result_auc_test must contain 'AUC' and 'AP' keys")

    # Return results
    return (
        mrr_pos2neg,
        mrr_neg2pos,
        result_auc_test,
        pos_edge_index_err,
        pos_rank_err,
        neg_edge_index_err,
        neg_rank_err
    )


def plot_rank_list(position: torch.tensor, label: str, sample_size: int = 20):
    """ Plot the ranks for each query in the dataset """
    pos = position.clone()

    index = np.random.randint(1, pos.shape[0], size=sample_size)
    pos = pos[index]
    # renormalize the ranks to be between 1 and 50
    pos = ((pos - pos.min()) * (50 - 1) / (pos.max() - pos.min())).to(torch.int32)

    # Create the plot
    plt.figure()
    fig, axes = plt.subplots(1, pos.shape[0], figsize=(20, 5))  # Adjust figsize for better visualization

    for i, ax in tqdm(enumerate(axes)):
        # Create a bar chart for each query
        ax.barh(range(int(pos.max())), [0.5] * int(pos.max()), color='lightgrey', height=0.5)
        ax.barh(pos[i], 0.5, color='lightcoral', height=0.5)
        ax.invert_yaxis()

    for ax in axes:
        ax.axis('off')

    # Random ID for the plot filename
    id = random.randint(1, 100)
    plt.savefig(f'plot_rank_{label}_{id}.png')


def find_opt_thres(pos_probs, neg_probs, thres=None):
    if thres is None:
        thres = np.linspace(0, 1, num=100)
    y_true = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    y_scores = np.concatenate([pos_probs, neg_probs])
    best_thres = 0
    best_acc = 0
    for t in thres:
        y_pred = (y_scores >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thres = t

    pos_pred = (pos_probs >= best_thres).astype(int)
    neg_pred = (neg_probs >= best_thres).astype(int)
    return best_thres, best_acc, pos_pred, neg_pred


def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_results(file_path):
    data = load_csv(file_path)

    if data is not None:
        print(data.head())  # Print the first few rows of the DataFrame

    pos_mask = data[data['gr'] == 1.0]
    pos_index = pos_mask[['edge_index0', 'edge_index1']]

    neg_mask = data[data['gr'] == 0.0]
    neg_index = neg_mask[['edge_index0', 'edge_index1']]
    P1 = pos_mask['pred'].to_numpy()
    P2 = neg_mask['pred'].to_numpy()
    neg_index = neg_index.to_numpy()
    pos_index = pos_index.to_numpy()

    return P1, P2, pos_index, neg_index


def plot_pos_neg_histogram(Pos, Neg, best_thres=None):
    plt.figure(figsize=(12, 8))
    # Plot distributions of probabilities
    plt.hist(Pos, bins=100, alpha=0.5, color='blue', label='Positive Class')
    plt.hist(Neg, bins=100, alpha=0.5, color='red', label='Negative Class')

    plt.axvline(best_thres, color='green', linestyle='--', label=f'Optimal Threshold = {best_thres:.2f}')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Probability Distributions with Optimal Threshold')
    plt.show()


def visual_adjacency(name, num_nodes, pos_edges, neg_edges, pos_weight, neg_weight):
    """
    Visualizes the adjacency matrix of a graph with positive and negative edges,
    using edge weights to determine color density.

    Parameters:
    - name (str): The name of the graph.
    - num_nodes (int): The number of nodes in the graph.
    - pos_edges (ndarray): Array of positive edges, each edge represented by [row, col].
    - neg_edges (ndarray): Array of negative edges, each edge represented by [row, col].
    - pos_weight (ndarray): Weights for positive edges, values between 0 and 1.
    - neg_weight (ndarray): Weights for negative edges, values between 0 and 1.
    """

    fig_size = (40, 40) if num_nodes > 100000 else (20, 20)
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_xlim(-0.5, num_nodes - 0.5)
    ax.set_ylim(-0.5, num_nodes - 0.5)
    ax.set_aspect('equal')

    pos_weight[pos_weight == 0] = 0.5
    neg_weight[neg_weight == 0] = 0.5

    scatter_pos = ax.scatter(pos_edges[:, 1], pos_edges[:, 0], color='blue', alpha=pos_weight, label='Positive Edges',
                             s=100)
    scatter_neg = ax.scatter(neg_edges[:, 1], neg_edges[:, 0], color='red', alpha=neg_weight, label='Negative Edges',
                             s=100)
    plt.title(f'Adjacency Matrix - {name}')
    plt.legend(loc='upper right')
    plt.savefig(f'Adjacency_Matrix_{name}.png')
    plt.close()


def shared_rows(pos_index_llama, neg_index_llama):
    set1 = set(map(tuple, pos_index_llama))
    set2 = set(map(tuple, neg_index_llama))
    shared_rows = set1.intersection(set2)
    shared_rows_array = np.array(list(shared_rows))
    print(f'There are {len(shared_rows)} shared rows between the two arrays.')
    return shared_rows_array


def tensor_to_csr_matrix(edge_index: torch.tensor):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    num_nodes = edge_index.max().item() + 1  # Assuming node indices are 0-based
    adj_coo = ssp.csr_matrix((np.ones(src.shape[0]), (src, dst)), shape=(num_nodes, num_nodes))
    return adj_coo


def eval_mix_heuristic(data, pos_edge_index_err):
    pos_edge_index_err = torch.as_tensor(pos_edge_index_err)
    heuristic_feat = []
    for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
        edge_index = tensor_to_csr_matrix(data.edge_index)
        scores, edge_index = eval(use_lsf)(edge_index, pos_edge_index_err.T)
        heuristic_feat.append(scores.numpy())

    for use_gsf in ['Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR']:
        edge_index = tensor_to_csr_matrix(data.edge_index)
        scores, edge_index = eval(use_gsf)(edge_index, pos_edge_index_err.T)
        heuristic_feat.append(scores.numpy())

    feat_pro = []
    for rows in pos_edge_index_err:
        feat_pro.append((data.x[rows[0]] @ data.x[rows[1]]).item())
    feat_pro = np.array(feat_pro)
    heuristic_feat.append(feat_pro)
    heuristic_feat = np.vstack(heuristic_feat)

    if len(heuristic_feat.shape) == 2:
        row_sums = heuristic_feat.sum(axis=1, keepdims=True)
        normalized_feat = heuristic_feat / row_sums

    return normalized_feat


def save_error_examples(type_2, text):
    data = []

    for row in type_2:
        src = text[row[0]]
        tgt = text[row[1]]
        data.append({'Source': src, 'Target': tgt})

        # Display each example (optional)
        # display(Markdown(f"**Source:** {src}  \n**Target:** {tgt}"))

    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    import os
    import sys
    from IPython.display import display, Markdown
    import pandas as pd
    from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
    import torch

    from matplotlib import pyplot as plt
    from core.data_utils.load import load_data_lp
    from core.graphgps.utility.utils import init_cfg_test

    # Assuming your target directory is one level up from the current working directory
    notebook_dir = os.getcwd()
    target_dir = os.path.abspath(os.path.join(notebook_dir, '..'))

    sys.path.insert(0, target_dir)

    cfg = init_cfg_test()
    splits, text, data = load_data_lp[cfg.data.name](cfg.data)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    FILE_PATH = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/educational_demo/'
    llama_cora_path = FILE_PATH + 'err_ncnc_llama/llama-cora-origin_dot_AUC_0.903_MRR_0.228.csv'

    P1_llama, P2_llama, pos_index_llama, neg_index_llama = load_results(llama_cora_path)
    best_thres_llama, best_acc_llama, pos_pred_llama, neg_pred_llama = find_opt_thres(P1_llama, P2_llama)

    plot_pos_neg_histogram(P1_llama, P2_llama, best_thres_llama)

    pos_weight = (pos_pred_llama == 1).astype(int)
    neg_weight = (neg_pred_llama == 0).astype(int)
    visual_adjacency('llama_cora', data.num_nodes, pos_index_llama, neg_index_llama, pos_weight, neg_weight)

    k_list = [0.1, 0.2, 0.3, 0.5, 1]
    mrr_pos2neg, mrr_neg2pos, result_auc_test, pos_edge_index_err, pos_rank_err, neg_edge_index_err, neg_rank_err = get_metric_invariant(
        torch.tensor(P1_llama), torch.tensor(pos_index_llama), torch.tensor(P2_llama), torch.tensor(neg_index_llama),
        k_list)

    ncnc_cora_path = FILE_PATH + 'err_ncnc_llama/ncnc-cora_AUC_0.9669_MRR_0.5275.csv'
    P1, P2, pos_index, neg_index = load_results(ncnc_cora_path)
    best_thres, best_acc, pos_pred, neg_pred = find_opt_thres(P1, P2)
    plot_pos_neg_histogram(P1, P2, best_thres)

    pos_weight = (pos_pred == 1).astype(int)
    neg_weight = (neg_pred == 0).astype(int)
    visual_adjacency('ncnc_cora', data.num_nodes, pos_index, neg_index, pos_weight, neg_weight)
    mrr_pos2neg, mrr_neg2pos, result_auc_test, pos_edge_index_err, pos_rank_err, neg_edge_index_err, neg_rank_err = get_metric_invariant(
        torch.tensor(P1), torch.tensor(pos_index), torch.tensor(P2), torch.tensor(neg_index), k_list)

    shared_rows(neg_index_llama, neg_index)
    shared_pos_error = shared_rows(pos_index_llama, pos_index)

    for row in shared_pos_error:
        src = text[row[0]]
        tgt = text[row[1]]
        print(f"{row[0]}**Source:** {src}  \n {row[1]}**Target:** {tgt}")
        display(Markdown(f"**Source:** {src}  \n**Target:** {tgt}"))


