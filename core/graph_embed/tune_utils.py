
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import csv
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN
from heuristic.eval import get_metric_score
from ogb.linkproppred import Evaluator
import argparse
import pandas as pd 
from typing import Dict
import copy

from graphgps.utility.utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel,
)

# Constants
FILE_PATH = get_git_repo_root_path() + '/'

set_float = lambda result: float(result.split(' ± ')[0])

def set_cfg(file_path, args):
    with open(file_path + args.cfg_file, "r") as f:
        return CN.load_cfg(f)

def load_sweep_config(file_path):
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)

def print_args(args):
    print(args)

def initialize_config(file_path, args):
    cfg = set_cfg(file_path, args)
    cfg.merge_from_list(args.opts)
    torch.set_num_threads(cfg.num_threads)
    
    return cfg

def save_parameters(root, model, start, end, epochs):
    file_path = root + '/model_parameters.csv'
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model Name", "Total num", "Time 1 epoch"])
        total_params = model.count_parameters()
        model_name = model.__class__.__name__
        writer.writerow([model_name, total_params, (end - start) / epochs])
# TODO how to save wandb files 

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/lms/tfidf.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gat_sp1.yaml',
                        help='The configuration file path.')
    
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()



def process_edge_index(full_edge_index):
    print("full_edge_index", full_edge_index.shape)
    return to_scipy_sparse_matrix(full_edge_index)


def process_value(v):
    return (lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)(v)


def train_and_evaluate_logistic_regression(id, X_train, y_train, X_test, y_test, max_iter, method):
    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='auto')
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print("acc", acc)

    y_pred = clf.predict_proba(X_test)
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    results_mrr = {'node2vec_mrr': result_mrr}

    root = FILE_PATH + 'results'
    acc_file = root + f'/{cfg.data.name}_acc.csv'
    mrr_file = root + f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    append_acc_to_excel(id, results_acc, acc_file, cfg.data.name, method)
    append_mrr_to_excel(id, results_mrr, mrr_file, method)

    print(results_acc, '\n', results_mrr)
    return acc

#param_tune_acc_mrr

def param_tune_acc_mrr(uuid_val, metrics, root, name, method):
    # if not exists save the first row
    
    # input processing 
    first_value_type = type(next(iter(metrics.values())))
    if all(isinstance(value, first_value_type) for value in metrics.values()):
        if first_value_type == str:
                _, metrics = convert_to_float(metrics)

        
    csv_columns = ['Metric'] + list(k for k in metrics) 
    # load old csv
    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)
    
    if type(Data.values[0][1]) == str:
        _, Data_float = df_str2float(Data)
    # set float form for Data
    
    # create new line 
    acc_lst = []
    
    for k, v in metrics.items():
        acc_lst.append(process_value(v))
        
    # merge with old lines, 
    v_lst = [f'{name}_{uuid_val}_{method}'] + acc_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    # best value
    highest_values = new_Data.apply(lambda column: max(column, default=None))
    # concat and save
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)
    upt_Data = pd.concat([new_Data, Best_df])
    upt_Data.to_csv(root,index=False)
    return upt_Data


def save_parmet_tune(name_tag, metrics, root):
    
    csv_columns = ['Metric'] + list(metrics)

    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)

    new_lst = [process_value(v) for k, v in metrics.items()]
    v_lst = [f'{name_tag}'] + new_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    # best value
    highest_values = {}
    for column in new_Data.columns:
        try:
            highest_values[column] = new_Data[column].max()
        except:
            highest_values[column] = None

    # concat and save
    Best_list = ['Best'] + pd.Series(highest_values).tolist()[1:]
    # print(Best_list)
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)

    upt_Data = pd.concat([new_Data, Best_df])
    upt_Data.to_csv(root,index=False)
    return upt_Data
    
    
def mvari_str2csv(name_tag, metrics, root):
    # if not exists save the first row
    # one for new string line 
    # another for new highest value line

    first_value_type = type(next(iter(metrics.values())))
    if all(isinstance(value, first_value_type) for value in metrics.values()):
        if first_value_type == str:
            metrics, float_metrics = convert_to_float(metrics)
        else:
            float_metrics = metrics

    new_df, csv_columns = dict2df(metrics, name_tag)
    new_df_float, csv_columns = dict2df(float_metrics, name_tag)
    
    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)

    new_lst = [process_value(v) for k, v in metrics.items()]
    v_lst = [f'{name_tag}'] + new_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    # best value
    highest_values = new_Data.apply(lambda column: max(column, default=None))
    # concat and save
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)

    upt_Data = pd.concat([new_Data, Best_df])
    upt_Data.to_csv(root,index=False)
    return upt_Data


def max_except_metric(column):
    if column.name == 'Metric':  
        return None# Check if the column is not named 'Metric'
    elif pd.api.types.is_numeric_dtype(column):  # Check if the column is numeric
            return column.max()
    else: 
        return None  # For non-numeric columns or 'Metric' column, return None


def dict2df(metrics: Dict[str, float], head: str) -> pd.DataFrame:
    csv_columns = ['Metric'] + list(k for k in metrics) 

    # create new line 
    acc_lst = []
    
    for _, v in metrics.items():
        acc_lst.append(process_value(v))
        
    # merge with old lines
    v_lst = [head] + acc_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    
    return new_df, csv_columns

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def df_str2float(df: pd.DataFrame) -> pd.DataFrame:
    df_float = copy.deepcopy(df)
    for index, row in df_float.iterrows():
        for column_name, value in row.items():
            if len(value.split('±')) == 1:
                continue
            elif is_float(value):
                value = float(value)
            else:
                df_float.at[index, column_name] = set_float(value)
    return df, df_float


def convert_to_float(metrics: Dict[str, str]) -> Dict[str, float]:
    float_metrics = copy.deepcopy(metrics)
    for key, val in float_metrics.items():
        float_metrics[key] = set_float(val)
    return metrics, float_metrics


def mvari_str2csv(name_tag, metrics, root):

    first_value_type = type(next(iter(metrics.values())))
    if all(isinstance(value, first_value_type) for value in metrics.values()):
        if first_value_type == str:
            metrics, float_metrics = convert_to_float(metrics)
        else:
            float_metrics = metrics

    new_df, csv_columns = dict2df(metrics, name_tag)
    new_df_float, csv_columns = dict2df(float_metrics, name_tag)
    
    try:
        Data = pd.read_csv(root)[:-1]
        Data, Data_float = df_str2float(Data)
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data, Data_float = df_str2float(Data)
        Data.to_csv(root, index=False)
    
    # debug
    new_Data = pd.concat([Data, new_df])
    new_Data_float = pd.concat([Data_float, new_df_float])
            
    # best value
    new_Data_float[new_Data_float.columns[1:]] = new_Data_float[new_Data_float.columns[1:]].astype(float)
    highest_values = new_Data_float.apply(lambda column: max(column, default=None))
            
    # concat and save
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)
    upt_Data = pd.concat([new_Data, Best_df])
    
    upt_Data.to_csv(root, index=False)
    print(f"result is saved to {root}.")
    return upt_Data