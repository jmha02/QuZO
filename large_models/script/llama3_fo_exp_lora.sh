# Zero-shot
export HF_HOME=/home/thomasjjc/resource_dir/huggingface
# export PATH=/home/thomasjjc/resource_dir/llm_quant/bin:$PATH
export LOG_HOME=/home/thomasjjc/resource_dir/llm_quant

pip install transformers==4.45.0
# Load 4/8-bit model throught bitandbytes lib
cd /home/thomasjjc/project/QuZO/large_models
# CUDA_VISIBLE_DEVICES=3 TAG=INT8model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=8 ABIT=8 BS=2 MODE=lora LR=1e-5 bash finetune.sh 
# pip install transformers==4.45.0
# CUDA_VISIBLE_DEVICES=3 TAG=INT8model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=8 ABIT=8 BS=2 MODE=lora LR=3e-5 bash finetune.sh 
# pip install transformers==4.45.0
# CUDA_VISIBLE_DEVICES=3 TAG=INT8model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=8 ABIT=8 BS=1 MODE=lora LR=1e-5 bash finetune.sh 
# CUDA_VISIBLE_DEVICES=3 TAG=INT8model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=8 ABIT=8 BS=2 MODE=lora LR=5e-5 bash finetune.sh 
# pip install transformers==4.45.0

CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=ReCoRD WBIT=4 ABIT=8 BS=2 MODE=lora LR=1e-5 bash finetune.sh 
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=SQuAD WBIT=4 ABIT=8 BS=2 MODE=lora LR=3e-5 bash finetune.sh 
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=DROP WBIT=4 ABIT=8 BS=1 MODE=lora LR=1e-5 bash finetune.sh 
CUDA_VISIBLE_DEVICES=3 TAG=INT4model-FO-QAT-LoRA MODEL=meta-llama/Meta-Llama-3-8B TASK=MultiRC WBIT=4 ABIT=8 BS=2 MODE=lora LR=5e-5 bash finetune.sh 
