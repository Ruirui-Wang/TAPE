import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import csv
from data_utils.load_data_nc import (load_tag_cora, 
                                     load_tag_pubmed, 
                                     load_tag_product, 
                                     load_tag_ogbn_arxiv,
                                     load_tag_product, 
                                     load_tag_arxiv23,
                                     load_tag_citeseer,
                                     load_tag_citationv8,
                                     load_tag_photo,
                                     load_tag_history,

                                     )
from data_utils.load_data_lp import (load_taglp_arxiv2023, 
                                    load_taglp_cora, 
                                    load_taglp_pubmed, 
                                    load_taglp_product, 
                                    load_taglp_ogbn_arxiv,
                                    load_taglp_citeseer,
                                     load_taglp_citationv8,
                                     load_taglp_pwc_small,
                                     load_taglp_pwc_large,
                                     load_taglp_pwc_medium,
                                     load_taglp_photo,
                                     load_taglp_history,
                                     )
from data_utils.load_data_lp import (load_graph_cora, 
                                     load_graph_arxiv23,
                                     load_graph_ogbn_arxiv,
                                     load_graph_pubmed,
                                     load_graph_citeseer,
                                     load_graph_citationv8,
                                     load_graph_pwc_small,
                                     load_graph_pwc_large,
                                     load_graph_pwc_medium,
                                     )

# TODO standarize the input and output
load_data_nc = {
    'cora': load_tag_cora,
    'pubmed': load_tag_pubmed,
    'arxiv_2023': load_tag_arxiv23,
    'ogbn-arxiv': load_tag_ogbn_arxiv,
    'ogbn-products': load_tag_product,
    'citeseer': load_tag_citeseer,
    'citationv8': load_tag_citationv8,
}

load_data_lp = {
    'cora': load_taglp_cora,
    'pubmed': load_taglp_pubmed,
    'arxiv_2023': load_taglp_arxiv2023,
    'ogbn-arxiv': load_taglp_ogbn_arxiv,
    'ogbn_products': load_taglp_product,
    'citeseer': load_taglp_citeseer,
    'citationv8': load_taglp_citationv8,
    'pwc_small': load_taglp_pwc_small,
    'pwc_large': load_taglp_pwc_large,
    'pwc_medium': load_taglp_pwc_medium,
    'photo': load_taglp_photo,
    'history': load_taglp_history,
}

load_graph_lp = {
    'cora': load_graph_cora,
    'pubmed': load_graph_pubmed,
    'arxiv_2023': load_graph_arxiv23,
    'ogbn-arxiv': load_graph_ogbn_arxiv,
    'citeseer': load_graph_citeseer,
    'citationv8': load_graph_citationv8,
    'pwc_small': load_graph_pwc_small,
    'pwc_large': load_graph_pwc_large,
    'pwc_medium': load_graph_pwc_medium,
}


def load_gpt_preds(dataset, topk):
    preds = []
    fn = f'gpt_preds/{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl

