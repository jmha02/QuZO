# Zero-shot
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
# export PATH=/home/thomasjjc/resource_dir/llm_quant/bin:$PATH
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant

pip install transformers==4.31.0

# Load 4/8-bit model throught bitandbytes lib
cd /home/thomasjjc/project/QuZO/large_models
CUDA_VISIBLE_DEVICES=2 TAG=QuZO-LPmodel-QFT-Ours MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=4 PBIT=4 MODE=qft BS=16 LR=2e-6 EPS=2e-3 TRAINER=zo_lowbit_ft TWO=True STEPS=6000 bash quzo.sh &
# CUDA_VISIBLE_DEVICES=1 TAG=QuZO-LPmodel-QFT-Ours MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=4 PBIT=4 MODE=qft BS=8 LR=2e-6 EPS=1e-3 TRAINER=zo_lowbit_ft TWO=True STEPS=6000 bash quzo.sh &

# CUDA_VISIBLE_DEVICES=1 TAG=QuZO-LPmodel-QFT-Ours MODEL=meta-llama/Llama-2-7b-hf TASK=SQuAD WBIT=4 PBIT=4 MODE=qft BS=4 LR=7e-7 EPS=1e-3 TRAINER=zo_lowbit_ft TWO=True STEPS=6000 bash quzo.sh &

