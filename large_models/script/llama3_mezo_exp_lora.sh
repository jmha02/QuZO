# Zero-shot
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
# export PATH=/home/thomasjjc/resource_dir/llm_quant/bin:$PATH
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant

pip install transformers==4.45.0
# Load 4/8-bit model throught bitandbytes lib
cd /home/thomasjjc/project/QuZO/large_models
# CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=8 PBIT=4 MODE=lora BS=4 LR=3e-5 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 
# CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=8 PBIT=4 MODE=lora BS=4 LR=1e-5 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 
# CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=8 PBIT=4 MODE=lora BS=8 LR=5e-5 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 
# CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=8 PBIT=4 MODE=lora BS=8 LR=1e-4 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 

CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=4 PBIT=4 MODE=lora BS=4 LR=3e-5 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 
CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=4 PBIT=4 MODE=lora BS=4 LR=1e-5 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 
CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 PBIT=4 MODE=lora BS=8 LR=5e-5 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 
CUDA_VISIBLE_DEVICES=1 TAG=MeZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=4 PBIT=4 MODE=lora BS=8 LR=1e-4 EPS=2e-3 TRAINER=zo TWO=True STEPS=6000 bash quzo.sh 


# CUDA_VISIBLE_DEVICES=1 TAG=QuZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=4 PBIT=4 MODE=lora BS=4 LR=1e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=True STEPS=6000 bash quzo.sh 
# CUDA_VISIBLE_DEVICES=1 TAG=QuZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=4 PBIT=4 MODE=lora BS=4 LR=3e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=True STEPS=6000 bash quzo.sh 
# CUDA_VISIBLE_DEVICES=1 TAG=QuZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 PBIT=4 MODE=lora BS=4 LR=5e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=True STEPS=6000 bash quzo.sh 
# CUDA_VISIBLE_DEVICES=1 TAG=QuZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=4 PBIT=4 MODE=lora BS=8 LR=1e-4 EPS=2e-3 TRAINER=zo_lowbit TWO=True STEPS=6000 bash quzo.sh 
# CUDA_VISIBLE_DEVICES=1 TAG=QuZO-LPmodel-LoRA-Ours MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 PBIT=4 MODE=lora BS=2 LR=5e-5 EPS=2e-3 TRAINER=zo_lowbit TWO=True STEPS=6000 bash quzo.sh 
