#!/bin/bash
# Script to evaluate existing STE checkpoints
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

# Evaluation command
python run_mezo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $CHECKPOINT_DIR \
    --run_name "$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT-EVAL" \
    --trainer regular \
    --use_ste \
    --ste_weight_bits $WBIT \
    --ste_activation_bits $ABIT \
    --ste_grad_bits $GBIT \
    --num_train 0 \
    --num_eval 1000 \
    --train_set_seed 0 \
    --no_eval false \
    --load_int4 true \
    --lora \
    --lora_alpha 16 \
    --lora_r 8

# For evaluating other tasks, uncomment and modify these:
# TASK=SQuAD
# CHECKPOINT_DIR=$LOG_HOME/$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT
# python run_mezo.py --model_name $MODEL --task_name $TASK --output_dir $CHECKPOINT_DIR --run_name "$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT-EVAL" --trainer regular --use_ste --ste_weight_bits $WBIT --ste_activation_bits $ABIT --ste_grad_bits $GBIT --num_train 0 --num_eval 1000 --train_set_seed 0 --no_eval false --load_int4 true --lora --lora_alpha 16 --lora_r 8

# TASK=DROP
# CHECKPOINT_DIR=$LOG_HOME/$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT
# python run_mezo.py --model_name $MODEL --task_name $TASK --output_dir $CHECKPOINT_DIR --run_name "$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT-EVAL" --trainer regular --use_ste --ste_weight_bits $WBIT --ste_activation_bits $ABIT --ste_grad_bits $GBIT --num_train 0 --num_eval 1000 --train_set_seed 0 --no_eval false --load_int4 true --lora --lora_alpha 16 --lora_r 8

# TASK=MultiRC
# CHECKPOINT_DIR=$LOG_HOME/$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT
# python run_mezo.py --model_name $MODEL --task_name $TASK --output_dir $CHECKPOINT_DIR --run_name "$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT-EVAL" --trainer regular --use_ste --ste_weight_bits $WBIT --ste_activation_bits $ABIT --ste_grad_bits $GBIT --num_train 0 --num_eval 1000 --train_set_seed 0 --no_eval false --load_int4 true --lora --lora_alpha 16 --lora_r 8 