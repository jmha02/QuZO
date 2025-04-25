# Zero-shot
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
# export PATH=/home/thomasjjc/resource_dir/llm_quant/bin:$PATH
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant

pip install transformers==4.31.0

# Load 4/8-bit model throught bitandbytes lib
cd /home/thomasjjc/project/QuZO/large_models
CUDA_VISIBLE_DEVICES=2 TAG=QuZO-LPmodel-LoRA-Q-RGE1 MODEL=meta-llama/Llama-2-7b-hf TASK=SQuAD WBIT=8 PBIT=4 MODE=lora BS=4 LR=4e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=False STEPS=6000 bash quzo.sh &
CUDA_VISIBLE_DEVICES=2 TAG=QuZO-LPmodel-LoRA-Ours MODEL=meta-llama/Llama-2-7b-hf TASK=SQuAD WBIT=8 PBIT=4 MODE=lora BS=4 LR=6e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=True STEPS=6000 bash quzo.sh &
