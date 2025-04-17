export HF_HOME=/global/cfs/cdirs/m4645/alvinliu/huggingface
export PATH=/global/homes/a/alvinliu/.conda/envs/qzo/bin/:$PATH
export LOG_HOME=/global/cfs/cdirs/m4645/alvinliu/qzo

TASK=${TASK:-SST2}

python run_debug.py --task_name $TASK
