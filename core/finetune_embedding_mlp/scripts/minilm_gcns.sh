CUDA_VISIBLE_DEVICES=3  WANDB_DISABLED=True python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/heart_gnn_models.yaml --model GIN_Variant
CUDA_VISIBLE_DEVICES=0  WANDB_DISABLED=True python core/finetune_embedding_mlp/lora_lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/ncn.yaml
CUDA_VISIBLE_DEVICES=2  WANDB_DISABLED=True python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/heart_gnn_models.yaml --model SAGE_Variant
CUDA_VISIBLE_DEVICES=0  WANDB_DISABLED=True python core/finetune_embedding_mlp/lora_lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/ncn.yaml

CUDA_VISIBLE_DEVICES=0  WANDB_DISABLED=True python core/finetune_embedding_mlp/qlora_lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml --decoder core/yamls/cora/gcns/ncn.yaml