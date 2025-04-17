# Quantized ZO training on Large Autoregressive Language Models

New FSDP Env based on 
```
requirement_fsdp.txt
```
## Installation
```
conda install  pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
transformers==4.28.1
accelerate==0.17.1

New version 0417:
conda create --name llm_qzo python=3.11
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.2 -c pytorch -c nvidia 
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia


transformers==4.31.0
accelerate==0.17.1
```
# Quantization CUDA kernel
```
gcc < 11
pip install ./quant
```


## Usage

Use `run.py` for all functions :
```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments. We introduce some of the most important ones below. 
* `--num_train`: Number of training examples. For ICL, this is the number of demonstrations.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot/ICL), `regular` (fine-tuning), `zo_lowbit` (QuZO), or `zo` (MeZO).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise,it is LM-style teacher forcing.
* `--zo_eps`: ZO hyperparameter epsilon.
* `--prefix_tuning`: use prefix-tuning. 
* `--Lora`: use LoRA.

We also support all [HuggingFace trainer arguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) for easily setting fine-tuning hyperparameters.

```bash



# FP32 FSDP worked (Verified)
NUM_GPU=$NUM_GPU TAG=Debug MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=32 ABIT=32 BS=8 PEFTMODE=ft QMODE=float LR=1e-5 EPS=1e-3 SAVE_STEPS=10 EPOCH=0.1 bash finetune_fsdp_llama.sh

# INT or FP Quant FSDP worked (Verified)
NUM_GPU=$NUM_GPU TAG=Debug MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=4 ABIT=8 BS=16 PEFTMODE=ft QMODE=float LR=1e-5 EPS=1e-3 SAVE_STEPS=10 EPOCH=5 bash finetune_fsdp_llama.sh
NUM_GPU=$NUM_GPU TAG=Debug MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=8 ABIT=8 BS=16 PEFTMODE=ft QMODE=float LR=1e-5 EPS=1e-3 SAVE_STEPS=10 EPOCH=5 bash finetune_fsdp_llama.sh
NUM_GPU=$NUM_GPU TAG=Debug MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=8 ABIT=8 BS=16 PEFTMODE=ft QMODE=int LR=1e-5 EPS=1e-3 SAVE_STEPS=10 EPOCH=5 bash finetune_fsdp_llama.sh
NUM_GPU=$NUM_GPU TAG=Debug MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=4 ABIT=8 BS=16 PEFTMODE=ft QMODE=int LR=1e-5 EPS=1e-3 SAVE_STEPS=10 EPOCH=5 bash finetune_fsdp_llama.sh


# QuZO (INT4,8 Experiments)

CUDA_VISIBLE_DEVICES=6 TAG=QAT-ZO-LPmodel-LoRA-Rank8-Ours MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 WBIT=8 PBIT=4 MODE=lora BS=16 LR=6e-5 EPS=1e-3 TRAINER=zo_lowbit TWO=False STEPS=8000 bash mezo.sh 
CUDA_VISIBLE_DEVICES=6 TAG=QAT-ZO-LPmodel-LoRA-Rank8-Ours MODEL=meta-llama/Llama-2-7b-hf TASK=BoolQ WBIT=8 PBIT=4 MODE=lora BS=8 LR=1e-6 EPS=1e-3 TRAINER=zo_lowbit TWO=True STEPS=10000 bash mezo.sh &
CUDA_VISIBLE_DEVICES=3 TAG=QAT-ZO-LPmodel-LoRA-Rank8-Ours MODEL=meta-llama/Llama-2-7b-hf TASK=RTE WBIT=4 PBIT=4 MODE=lora BS=16 LR=5e-4 EPS=1e-3 TRAINER=zo_lowbit TWO=True STEPS=10000 bash mezo.sh 

```

