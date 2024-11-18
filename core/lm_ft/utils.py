import os
import numpy as np
import time
import datetime
import pytz

import subprocess as sp
from pathlib import Path
from types import SimpleNamespace as SN
import random, torch 

LINUX_HOME = str(Path.home())

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ServerInfo:
    def __init__(self):
        self.gpu_mem, self.gpus, self.n_gpus = 0, [], 0
        try:
            import numpy as np
            command = "nvidia-smi --query-gpu=memory.total --format=csv"
            gpus = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            self.gpus = np.array(range(len(gpus)))
            self.n_gpus = len(gpus)
            self.gpu_mem = round(int(gpus[0].split()[0]) / 1024)
            self.sv_type = f'{self.gpu_mem}Gx{self.n_gpus}'
        except:
            print('NVIDIA-GPU not found, set to CPU.')
            self.sv_type = f'CPU'

    def __str__(self):
        return f'SERVER INFO: {self.sv_type}'


SV_INFO = ServerInfo()

PROJ_NAME = 'TAG-Benchmark'
# ! Project Path Settings

GPU_CF = {
    'py_path': f'{str(Path.home())}/miniconda/envs/ct/bin/python',
    'mnt_dir': f'{LINUX_HOME}/{PROJ_NAME}/',
    'default_gpu': '0',
}
CPU_CF = {
    'py_path': f'python',
    'mnt_dir': '',
    'default_gpu': '-1',
}
get_info_by_sv_type = lambda attr, t: CPU_CF[attr] if t == 'CPU' else GPU_CF[attr]
DEFAULT_GPU = get_info_by_sv_type('default_gpu', SV_INFO)
PYTHON = get_info_by_sv_type('py_path', SV_INFO)
# MNT_DIR = get_info_by_sv_type('mnt_dir', SV_INFO)

import os.path as osp

PROJ_DIR = osp.abspath(osp.dirname(__file__)).split('LMs')[0]
LM_PROJ_DIR = osp.join(PROJ_DIR, 'LMs/')


MNT_DIR = PROJ_DIR
# Temp paths: discarded when container is destroyed
TEMP_DIR = LM_PROJ_DIR
TEMP_PATH = f'{LM_PROJ_DIR}temp/'
LOG_PATH = f'{LM_PROJ_DIR}log/'

MNT_TEMP_DIR = f'{MNT_DIR}temp/'
TEMP_RES_PATH = f'{LM_PROJ_DIR}temp_results/'
RES_PATH = f'{LM_PROJ_DIR}results/'
DB_PATH = f'{LM_PROJ_DIR}exp_db/'

# ! Data Settings
DATA_PATH = f'{MNT_DIR}data/'
OGB_ROOT = f'{MNT_DIR}data/ogb/'
AMAZON_ROOT = f'{MNT_DIR}data/CSTAG/'
# DBLP_ROOT = f'{MNT_DIR}data/dblp/'
DBLP_ROOT = f'{MNT_DIR}data/CSTAG/'
GOOD_ROOT = f'{MNT_DIR}data/good/'
WEBKB_ROOT = f'{MNT_DIR}data/webkb/'

DATA_INFO = {
    'arxiv': {
        'type': 'ogb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 40,
        'n_nodes': 169343,
        'ogb_name': 'ogbn-arxiv',
        'raw_data_path': OGB_ROOT,  # Place to save raw data
        'max_length': 512,  # Place to save raw data ARXIV_ta 512； arxiv_T 64
        'data_root': f'{OGB_ROOT}ogbn_arxiv/',  # Default ogb download target path
        'raw_text_url': 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',
    },
    'Children': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 24,
        'n_nodes': 76875,
        'data_name': 'Books-Children',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Books/Children/',  # Default ogb download target path
    },
    'History': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 13,
        'n_nodes': 41551,
        'data_name': 'Books-History',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Books/History/',  # Default ogb download target path
    },
    'Computers': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 10,
        'n_nodes': 87229,
        'data_name': 'Computers',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Computers/',
    },
    'Fitness': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 13,
        'n_nodes': 173055,
        'data_name': 'Sports-Fitness',
        'max_length': 64,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Sports/Fit/',
    },
    'Photo': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 12,
        'n_nodes': 48362,
        'data_name': 'Electronics-Photo',
        'max_length': 512,  # Place to save raw data
        'train_year': 2015,
        'val_year': 2016,
        'data_root': f'{AMAZON_ROOT}Electronics/Photo/',
    },
    'Music': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 0,
        'n_nodes': 4290,
        'data_name': 'Digital-Music',
        'max_length': 40,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Digital/Music/',
    },
    'DBLP': {
        'type': 'dblp',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 40,
        'n_nodes': 1106759,
        # 'data_name': 'Citation-2015',
        'data_name': 'CitationV8',
        'max_length': 256,  # Place to save raw data
        # 'data_root': f'{DBLP_ROOT}Citation2015/',
        'data_root': f'{DBLP_ROOT}CitationV8/',
    },
    'Good': {
        'type': 'good',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 0,
        'n_nodes': 676084,
        'data_name': 'GoodReads',
        'max_length': 24,  # Place to save raw data
        'data_root': f'{GOOD_ROOT}Goodreads/',
    },
    'Cornell': {
        'type': 'webkb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 5,
        'n_nodes': 191,
        'data_name': 'Cornell',
        'max_length': 256, # Place to save raw data
        'data_root': f'{WEBKB_ROOT}Cornell/',
    },
    'Texas': {
        'type': 'webkb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 5,
        'n_nodes': 187,
        'data_name': 'Texas',
        'max_length': 256, # Place to save raw data
        'data_root': f'{WEBKB_ROOT}Texas/',
    },
    'Washington': {
        'type': 'webkb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 5,
        'n_nodes': 229,
        'data_name': 'Washington',
        'max_length': 256, # Place to save raw data
        'data_root': f'{WEBKB_ROOT}Washington/',
    },
    'Wisconsin': {
        'type': 'webkb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 5,
        'n_nodes': 265,
        'data_name': 'Wisconsin',
        'max_length': 256, # Place to save raw data
        'data_root': f'{WEBKB_ROOT}Wisconsin/',
    }
    ,
    'CitationV8': { 
        'type': 'good',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 0,
        'n_nodes': 1106759,
        'data_name': 'GoodReads',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{GOOD_ROOT}Goodreads/',
    }
}

get_d_info = lambda x: DATA_INFO[x.split('_')[0]]

DATASETS = list(DATA_INFO.keys())
DEFAULT_DATASET =  'History_DT' #'Children_TB'#'arxiv_TA'
DEFAULT_D_INFO = get_d_info(DEFAULT_DATASET)

# Datasets Name
# arxiv_TA/ Children_DT / History_DT/ Fitness_T / Computers_RS / Photo_RS / Music_T/


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper
