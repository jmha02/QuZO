# QuZO: Quantized Zero-Order Optimization for Large Language Models

## Environment Setup

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

# Quantized ZO Fine-Tuning Language Models 


QuZO is the first framework designed for efficient fine-tuning of quantized Large Language Models (LLMs) using low-bit stochastic perturbations and zeroth-order optimization (ZO). QuZO eliminates the need for backpropagation and achieves state-of-the-art performance in memory-constrained settings by leveraging inference-only engines for optimization.

🚀 Key Features

	•Quantized Fine-Tuning: Supports fine-tuning of 4-bit and 8-bit quantized LLMs, such as LLaMA-2 and OPT models.
	•Forward-Pass Only Training: Eliminates backward passes and reduces memory overhead by up to 5.46× compared to first-order (FO) methods like QLoRA.
	•Low-Bit Stochastic Perturbations: Employs quantized perturbations to estimate gradients efficiently.
	•Parameter-Efficient Fine-Tuning (PEFT): Compatible with LoRA and similar methods, focusing updates on low-rank matrices to enhance hardware efficiency.
	•Scalable and Flexible: Suitable for various model scales (from medium models like RoBERTa to large-scale LLaMA-2 models).

🛠 Installation

Clone the repository and set up the environment following the large_models/medium_models folder



🔧 Usage

1. Fine-Tuning with QuZO

To fine-tune a quantized model using QuZO, follow these steps in the large_models folder

2. Evaluate the Model

Evaluate the fine-tuned model:

To reproduce RoBERTa-large experiments, please refer to the medium_models folder. For LLaMa-2 experiments, please take a look at the large_models folder. 

📝 Framework Highlights

• Mixed-Precision Training

QuZO supports hybrid formats (e.g., FP8 activations with INT4 weights) to balance representation range and quantization errors.

• Parameter Efficiency with LoRA

QuZO integrates with LoRA, updating only the low-rank matrices ￼ and ￼ for fine-tuning, reducing trainable parameters significantly.










