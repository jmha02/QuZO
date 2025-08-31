#!/bin/bash
# Script to evaluate all STE tasks from llama3_ste_exp_lora.sh
export HF_HOME=/home/jjc/resource_dir/huggingface
# export PATH=/home/thomasjjc/resource_dir/llm_quant/bin:$PATH
export LOG_HOME=/home/jjc/resource_dir/llm_quant

cd /home/jjc/project/QuZO/large_models

# Common variables
MODEL=mistralai/Mistral-7B-Instruct-v0.3
WBIT=4
ABIT=8
GBIT=4
TAG=INT4model-STE-QAT-LoRA

# Function to evaluate a single task
evaluate_task() {
    local TASK=$1
    local BS=$2
    local LR=$3
    
    echo "Evaluating $TASK..."
    
    # Path to checkpoint directory
    CHECKPOINT_DIR=$LOG_HOME/$TAG-$TASK-Meta-Llama-3-8B-WBIT$WBIT-ABIT$ABIT
    
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
        --load_int8 false \
        --lora \
        --lora_alpha 16 \
        --lora_r 8 \
        --resume_from_checkpoint $CHECKPOINT_DIR
    
    echo "Evaluation of $TASK completed."
    echo "-----------------------------------------"
}

# Evaluate all four tasks with their specific parameters
# Parameters match those in llama3_ste_exp_lora.sh
echo "Starting evaluation of all STE tasks..."

# Evaluate ReCoRD
evaluate_task "ReCoRD" 2 "1e-5"

# # Evaluate SQuAD
evaluate_task "SQuAD" 2 "3e-5"

# Evaluate DROP
evaluate_task "DROP" 1 "1e-5"

# # Evaluate MultiRC
evaluate_task "MultiRC" 2 "5e-5"

echo "All evaluations completed!" 