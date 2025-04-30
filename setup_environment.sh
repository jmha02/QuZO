#!/bin/bash

# Print colorful messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up QuZO environment...${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda not found. Please install Conda first.${NC}"
    exit 1
fi

# Create environment from YAML
echo -e "${GREEN}Creating conda environment from quzo_environment.yml...${NC}"
conda env create -f quzo_environment.yml

# Activate environment
echo -e "${GREEN}Activating llm_quzo environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate llm_quzo

# Install quantization CUDA kernel
echo -e "${GREEN}Installing quantization CUDA kernel...${NC}"
cd large_models
pip install ./quant

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}You can activate the environment with:${NC}"
echo -e "${BLUE}conda activate llm_quzo${NC}"
echo -e "${BLUE}============================================${NC}" 