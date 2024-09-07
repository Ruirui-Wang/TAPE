import os, sys
sys.path.insert(0, '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/core')

import numpy as np
from graph_embed.ge.classify import read_node_label, Classifier
from graph_embed.ge.models.struc2vec import Struc2Vec
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from core.graphgps.utility.utils import (
    get_git_repo_root_path
)
from graph_embed.tune_utils import param_tune_acc_mrr
import uuid 
import wandb

FILE_PATH = get_git_repo_root_path() + '/'


def evaluate_embeddings(embeddings):

    X, Y = read_node_label('core/Embedding/data/flight/labels-brazil-airports.txt',skip_head=True)

    tr_frac = 0.8

    print("Training classifier using {:.2f}% nodes...".format(

        tr_frac * 100))

    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())

    evaluate = clf.split_train_evaluate(X, Y, tr_frac)
    
    return evaluate['acc']



def plot_embeddings(embeddings,):

    X, Y = read_node_label('core/Embedding/data/flight/labels-brazil-airports.txt',skip_head=True)
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    G = nx.read_edgelist('core/Embedding/data/flight/brazil-airports.edgelist', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])

    metrics = {}
    for wl  in [10, 15, 20]:
        for nw in [20, 40, 80]:
            for es in [16, 32, 64]:
                for ws in [5, 7, 9]:
                    model = Struc2Vec(G, 
                                    walk_length=wl, 
                                    num_walks=nw, 
                                    workers=es, 
                                    verbose=0, 
                                    data='flight',
                                    temp_path=f'./temp_path',
                                    reuse=False)
                    
                    model.train(embed_size=es, 
                                window_size=ws,
                                workers=20)
                    
                    embeddings = model.get_embeddings()

                    print(embeddings.shape)
                    # acc = evaluate_embeddings(embeddings)
                    # metrics.update({'wl': wl, 'nw': nw, 'es': es, 'ws': ws, 'acc': acc})
                    # root = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/results/flight.csv'
                    
                    # id = wandb.util.generate_id()
                    # param_tune_acc_mrr(id, metrics, root, 'flight', 'struc2vec')
                    # demo plot_embeddings(embeddings)
                    