
import numpy as np


from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from IPython import embed
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.graphgps.utility.utils import get_git_repo_root_path
from ge.classify import read_node_label, Classifier
from ge import Node2Vec

FILE_PATH = get_git_repo_root_path() + '/'

def evaluate_embeddings(embeddings, label_path):
    label_path = f'{FILE_PATH}dataset/data_embed/wiki/wiki_labels.txt'
    X, Y = read_node_label(label_path)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings, label_path):

    X, Y = read_node_label(label_path)

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
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.savefig('embeddings_wiki.png')


if __name__ == "__main__":
    
    G = nx.read_edgelist(f'{FILE_PATH}dataset/data_embed/wiki/Wiki_edgelist.txt',
                         create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])
    
    
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    
    model.train(embed_size=64, window_size = 5, iter = 3)
    embeddings=model.get_embeddings()
    print(embeddings)
    
    label_path = f'{FILE_PATH}dataset/data_embed/wiki/wiki_labels.txt'
    evaluate_embeddings(embeddings, label_path)
    plot_embeddings(embeddings, label_path)
    
