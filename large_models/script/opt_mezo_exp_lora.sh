# Zero-shot
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
# export PATH=/home/thomasjjc/resource_dir/llm_quant/bin:$PATH
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant



cd /home/thomasjjc/project/QuZO/large_models

# Load 4/8-bit model throught bitandbytes lib
CUDA_VISIBLE_DEVICES=0 TAG=MeZO-LPmodel-LoRA MODEL=facebook/opt-1.3b TASK=SST2 WBIT=8 PBIT=1 MODE=lora QMODE=int BS=16 LR=5e-5 EPS=1e-3 Trainer=zo STEPS=4000 bash mezo.sh &
# CUDA_VISIBLE_DEVICES=0 TAG=MeZO-LPmodel-LoRA MODEL=facebook/opt-1.3b TASK=SST2 WBIT=4 PBIT=1 MODE=lora QMODE=int BS=16 LR=5e-6 EPS=1e-3 Trainer=zo STEPS=4000 bash quzo.sh &
# CUDA_VISIBLE_DEVICES=0 TAG=MeZO-LPmodel-LoRA MODEL=facebook/opt-1.3b TASK=SST2 WBIT=32 PBIT=1 MODE=lora QMODE=int BS=16 LR=5e-6 EPS=1e-3 Trainer=zo STEPS=4000 bash quzo.sh &
