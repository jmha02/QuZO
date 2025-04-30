#!/bin/bash

# Default parameter values
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

TASK=${TASK:-SST2}
WBIT=${WBIT:-8}  # Weight quantization bits
ABIT=${ABIT:-8}  # Activation quantization bits
GBIT=${GBIT:-8}  # Gradient quantization bits
BS=${BS:-16}
MODE=${MODE:-ft}
PEFTMODE=${PEFTMODE:-lora}  # lora or ft
LR=${LR:-1e-5}
TAG=${TAG:-STE-FT}
LOG_HOME=${LOG_HOME:-/home/thomasjjc/resource_dir/llm_quant}
OUTPUT_DIR=${OUTPUT_DIR:-$LOG_HOME/$TAG-$TASK-$MODEL_NAME-WBIT$WBIT-ABIT$ABIT}
TRAINER=${TRAINER:-regular}

EXTRA_ARGS=""
TASK_ARGS=""

# Add LoRA if specified
if [ "$PEFTMODE" == "lora" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --lora --lora_alpha 16 --lora_r 8"
fi

# Handle weight bit precision loading
if [ "$WBIT" -eq 32 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_int8 False"
elif [ "$WBIT" -eq 8 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_int8 True"
elif [ "$WBIT" -eq 4 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_int4 True"
fi

# Add task-specific arguments
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    MultiRC) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA"
        ;;
    ReCoRD) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    DROP) # Can only fit real bsz = 1 on 80G A100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

# Create run name
RUN_NAME="$TAG-$TASK-$MODEL_NAME-WBIT$WBIT-ABIT$ABIT"

echo "Run name: $RUN_NAME"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Mode: $MODE"
echo "PEFT Mode: $PEFTMODE"
echo "Bits: W=$WBIT A=$ABIT G=$GBIT"
echo "Batch size: $BS"
echo "Learning rate: $LR"
echo "Output directory: $OUTPUT_DIR"
echo "Trainer: $TRAINER"
echo "Extra args: $EXTRA_ARGS"
echo "Task args: $TASK_ARGS"

# Run training using run_mezo.py instead of run.py
WANDB_PROJECT=${WANDB_PROJECT:-STE-Quant} python run_mezo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --trainer $TRAINER \
    --use_ste \
    --ste_weight_bits $WBIT \
    --ste_activation_bits $ABIT \
    --ste_grad_bits $GBIT \
    --train_as_classification \
    --per_device_train_batch_size $BS \
    --learning_rate $LR \
    --num_train_epochs 5 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" 