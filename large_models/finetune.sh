MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

EPOCH=${EPOCH:-5}
BS=${BS:-8}
LR=${LR:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
elif [ "$MODE" == "loretta_rep" ]; then
    TYPE="loretta_rep"
fi


TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
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

echo $TAG
echo "EPOCH: $EPOCH"
echo "BS: $BS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "WBIT: $WBIT"
echo "ABIT: $ABIT"
echo "PBIT: $PBIT"
echo "TWO: $TWO"


RUN_NAME="$TASK-${MODEL_NAME}-$REAL_BS-$LR"
if [ ! -z "$TAG" ]
then
    RUN_NAME="$TAG-$RUN_NAME-$REAL_BS-$LR-W$WBIT"
fi
if [ ! -z "$WBIT" ]
then
    RUN_NAME="$RUN_NAME-W$WBIT"
fi

if [ "$WBIT" -eq 32 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_int8 False"
elif [ "$WBIT" -eq 8 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_int8 True"
elif [ "$WBIT" -eq 4 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_int4 True"
fi

WANDB_PROJECT=${WANDB_PROJECT:-LLM_QAT_Perturb}

WANDB_PROJECT=$WANDB_PROJECT python run_mezo.py \
    --model_name $MODEL --run_name $RUN_NAME\
    --task_name $TASK  \
    --output_dir $LOG_HOME/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --trainer regular --fp16 False --optimizer sgd\
    --learning_rate $LR --num_train_epochs $EPOCH --per_device_train_batch_size $BS \
    --load_best_model_at_end False --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 \
    --train_as_classification \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"