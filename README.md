# TAG4LP
TAG4LP is a project to share the public text-attributed graph (TAG) datasets and benchmark the performance of the different baseline methods for link prediction
We welcome more to share datasets that are valuable for TAGs research.


## Datasets 🔔
We collect and construct 8 TAG datasets from Cora, Pubmed, Arxiv\_2023, ogbn-paper100M, citationv8, paperwithcode API.
Now you can go to the 'Files and version' in [TAG4LP](https://drive.google.com/file/d/15ZWzRESVpNFowt3zfm3v8-5DGdnMjFzk/view?usp=drive_link) to find the datasets we upload! 
In each dataset folder, you can find the **csv** file (which save the text attribute of the dataset), **pt** file (which represent the pyg graph file).
You can use the node initial feature we created, and you also can extract the node feature from our code under core/data_utils. 


## Environments
You can quickly install the corresponding dependencies
```shell
conda env create -f environment.yml
```

## Pipeline 🎮
We describe below how to use our repository to perform the experiments reported in the paper. We are also adjusting the style of the repository to make it easier to use.
(Please complete the ['Datasets'](get-tapedataset.sh) above first)
### 1. LMGJoint for Link Prediction
You can use Pwc_small Cora PubMed Arxiv_2023 Pwc_medium Ogbn_arxiv Citationv8 ogbn-papers100M.

```python
WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml
```

### 2. Reconstruct Baseline for PLM-related work 
```
WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-llama.yaml
```

## Config
all configuration file can be found be in core/yamls

