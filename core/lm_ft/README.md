## Fine Tune for Llama
WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml

python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml
