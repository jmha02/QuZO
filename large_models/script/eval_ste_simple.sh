#!/bin/bash
# Simplified script to evaluate existing STE checkpoints
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant

cd /home/thomasjjc/project/QuZO/large_models

# Variables
MODEL=meta-llama/Meta-Llama-3-8B
TASK=ReCoRD  # Change to the task you want to evaluate
WBIT=4
ABIT=8
GBIT=8
TAG=INT4model-STE-QAT-LoRA

# Path to checkpoint directory - change this to your checkpoint directory
CHECKPOINT_DIR=$LOG_HOME/$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT

# This approach avoids distributed training issues by:
# 1. Setting trainer to "none" so it doesn't try to train
# 2. Still loading the LoRA/STE weights from the checkpoint directory
# 3. Using num_train > 0 and train_set_seed to stay in the right branch of code
# 4. Setting a path to a different output directory to not overwrite 

# Evaluation command
CUDA_VISIBLE_DEVICES=0 python run_mezo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir "${CHECKPOINT_DIR}_eval" \
    --run_name "$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT-EVAL" \
    --trainer none \
    --use_ste \
    --ste_weight_bits $WBIT \
    --ste_activation_bits $ABIT \
    --ste_grad_bits $GBIT \
    --num_train 1 \
    --num_eval 1000 \
    --train_set_seed 0 \
    --load_int4 true \
    --lora \
    --lora_alpha 16 \
    --lora_r 8 \
    --resume_from_checkpoint $CHECKPOINT_DIR 