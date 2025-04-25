# Zero-shot
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
# export PATH=/home/thomasjjc/resource_dir/llm_quant/bin:$PATH
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant


# Load 4/8-bit model throught bitandbytes lib

cd /home/thomasjjc/project/QuZO/large_models
CUDA_VISIBLE_DEVICES=0 TAG=QuZO-LPmodel-LoRA-Ours MODEL=facebook/opt-1.3b TASK=DROP WBIT=4 PBIT=4 MODE=lora BS=4 LR=3e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=True STEPS=7000 bash quzo.sh &
CUDA_VISIBLE_DEVICES=0 TAG=QuZO-LPmodel-LoRA-Q-RGE1 MODEL=facebook/opt-1.3b TASK=DROP WBIT=4 PBIT=4 MODE=lora BS=4 LR=3e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=False STEPS=7000 bash quzo.sh &

