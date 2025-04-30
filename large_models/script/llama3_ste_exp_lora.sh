#!/bin/bash
# STE quantization experiment with 4-bit weights
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant

pip install transformers==4.45.0
cd /home/thomasjjc/project/QuZO/large_models

# 4-bit weight quantization experiments with different tasks
# Using Meta-Llama-3-8B model with STE quantization
# Weight bits = 4, Activation bits = 8, Gradient bits = 8

# Run experiments for each task (task-specific parameters handled automatically in finetune_ste.sh)
# Add train_set_seed=None explicitly to avoid assertion error
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=4 ABIT=8 GBIT=8 BS=2 MODE=lora LR=1e-5 PEFTMODE=lora TRAINER=regular bash finetune_ste.sh --num_train 1000 --num_dev 500 --num_eval 1000 --train_set_seed 0
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=4 ABIT=8 GBIT=8 BS=2 MODE=lora LR=3e-5 PEFTMODE=lora TRAINER=regular bash finetune_ste.sh --num_train 1000 --num_dev 500 --num_eval 1000 --train_set_seed 0
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=4 ABIT=8 GBIT=8 BS=1 MODE=lora LR=1e-5 PEFTMODE=lora TRAINER=regular bash finetune_ste.sh --num_train 1000 --num_dev 500 --num_eval 1000 --train_set_seed 0
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 ABIT=8 GBIT=8 BS=2 MODE=lora LR=5e-5 PEFTMODE=lora TRAINER=regular bash finetune_ste.sh --num_train 1000 --num_dev 500 --num_eval 1000 --train_set_seed 0

# Evaluation of trained models
echo "Evaluating trained models..."

# Evaluate ReCoRD
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=4 ABIT=8 GBIT=8 NUM_TRAIN=1 NUM_EVAL=1000 TRAIN_SET_SEED=0 bash script/eval_ste_simple.sh

# Evaluate SQuAD
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=4 ABIT=8 GBIT=8 NUM_TRAIN=1 NUM_EVAL=1000 TRAIN_SET_SEED=0 bash script/eval_ste_simple.sh

# Evaluate DROP
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=4 ABIT=8 GBIT=8 NUM_TRAIN=1 NUM_EVAL=1000 TRAIN_SET_SEED=0 bash script/eval_ste_simple.sh

# Evaluate MultiRC
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 ABIT=8 GBIT=8 NUM_TRAIN=1 NUM_EVAL=1000 TRAIN_SET_SEED=0 bash script/eval_ste_simple.sh

echo "All evaluations completed!"

# Uncomment to run additional tasks:
# CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=4 ABIT=8 GBIT=8 BS=2 MODE=lora LR=3e-5 PEFTMODE=lora TRAINER=regular bash finetune_ste.sh
# CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=4 ABIT=8 GBIT=8 BS=1 MODE=lora LR=1e-5 PEFTMODE=lora TRAINER=regular bash finetune_ste.sh
# CUDA_VISIBLE_DEVICES=3 TAG=INT4model-STE-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 ABIT=8 GBIT=8 BS=2 MODE=lora LR=5e-5 PEFTMODE=lora TRAINER=regular bash finetune_ste.sh

# CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=4 ABIT=8 BS=2 MODE=lora LR=1e-5 bash finetune.sh 
# CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=4 ABIT=8 BS=2 MODE=lora LR=3e-5 bash finetune.sh 
# CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=4 ABIT=8 BS=1 MODE=lora LR=1e-5 bash finetune.sh 
# CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 ABIT=8 BS=2 MODE=lora LR=5e-5 bash finetune.sh
