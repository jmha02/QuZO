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










