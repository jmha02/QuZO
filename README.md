# Source Code for Paper:  
**QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models**  
_Submitted to EMNLP 2025_  

**Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao,Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, Zheng Zhang**

---

QuZO introduces a **memory-efficient fine-tuning framework** for large language models (LLMs), leveraging **quantized zeroth-order (ZO) optimization** that eliminates gradient backpropagation and reduces memory footprint to near inference-level costs. Our approach enables **LLM adaptation on a single GPU**, even for models like LLaMA2-13B and OPT-30B, while maintaining competitive or superior accuracy compared to first-order and FP32-based ZO baselines (e.g., MeZO).

Key features include:

- **Memory Scalability:** Memory cost is bounded by the largest weight matrix instead of full model or activation footprint.
- **Single-GPU Compatibility:** Supports training 13B–30B models on a single A100 40GB GPU.
- **Efficient forward-only training:** Efficient per-step performance, supporting low-bit perturbations, sparse gradients, and adaptive query strategies.

---

##  Quickstart

Install the QuZO library with:

### Using Conda
You can easily set up the required environment using the provided YAML file:
```bash
# Clone the repository
git clone https://github.com/yourusername/QuZO.git
cd QuZO

# Create and activate the conda environment
conda env create -f llm_quzo_environment.yml
conda activate llm_quzo

# Install the quantization CUDA kernel
cd large_models
pip install ./quant
```

### Manual Installation
Alternatively, you can set up the environment manually:
```bash
# Create conda environment
conda create --name llm_quzo python=3.11
conda activate llm_quzo

# Install PyTorch and CUDA
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install transformers==4.31.0 accelerate==0.26.0 sentencepiece bitsandbytes==0.45.0 peft==0.14.0
pip install wandb datasets numpy==1.24.4 scikit-learn scipy tqdm

# Install quantization CUDA kernel
cd large_models
pip install ./quant
```


To run:

```bash
cd /home/jjc/project/QuZO/large_models/script
bash llama3_quzo_exp_lora.sh   
```

Model support
---
QuZO has been tested on:

- **LLaMA2/3**
- **OPT (1.3B/6.7B)**
- **Mistral (7B)**
- **RoBERTa / DeBERTa for classification tasks**

It supports common fine-tuning tasks including `CAUSAL_LM`, `SEQ_2_SEQ_LM`, and `SEQ_CLS`. Check our config docs for model-specific integration.


Environment
---
Generally, the package is implemented based on `torch==2.1.2`, `python=3.10.13` and `transformers==4.38.2`. For a detailed
list of environments we use, check `requirements.txt` or `environment.yml` files we provided.

Examples of reproducing the results in the paper
---
We provide one detailed example to reproduce the experimental results in our paper, which are stored in folder  `large_models`. To reproduce our experiments, follow the instructions below:

- Create the environment with the provided file `requirements.txt` or `environment.yml`
- setup the parameters in `run_all_bert_exp.sh` or `run_all_large_exp.sh` in each folder, which mainly contains:
  - MODEL: the name of the huggingface-enabled model
  - TASK: the name of the datasets, which support `MNLI, SST2, COLA, QQP, QNLI, RTE, MRPC, STSB` for bert tests and 
  `SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP` for llama tests
  - EPOCH/BS/LR: basic training argument for epochs, batch_size and learning_rate
  - DEVICE: the number of CUDA devices you would like to use `export CUDA_VISIBLE_DEVICES=$DEVICE`
  - For other arguments needed for the experiments, see `quzo.sh` for detail

Cite our paper
---
Note: The code is implemented based on an elder version of the [PEFT library](https://github.com/huggingface/peft/tree/main)

To use Loretta in your publication, please cite it by using the following BibTeX entry.
```angular2html
@article{zhou2025quzo,
  title={QuZO: Quantized zeroth-order fine-tuning for large language models},
  author={Zhou, Jiajun and Yang, Yifan and Zhen, Kai and Liu, Ziyue and Zhao, Yequan and Banijamali, Ershad and Mouchtaris, Athanasios and Wong, Ngai and Zhang, Zheng},
  journal={arXiv preprint arXiv:2502.12346},
  year={2025}
}
```
