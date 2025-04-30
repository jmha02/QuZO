#!/bin/bash

# Print colorful messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Fixing PEFT compatibility issues in QuZO...${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda not found. Please install Conda first.${NC}"
    exit 1
fi

# Activate the environment
echo -e "${GREEN}Activating llm_quzo environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate llm_quzo

# Install compatible PEFT version
echo -e "${GREEN}Installing compatible PEFT version (0.5.0)...${NC}"
pip install peft==0.5.0 --force-reinstall

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}PEFT library has been updated!${NC}"
echo -e "${BLUE}============================================${NC}"

# Define the context manager code to add to trainer files
CONTEXT_MANAGER_CODE='
@contextlib.contextmanager
def maybe_no_sync(model):
    """
    Context manager to handle models with or without no_sync method (used in distributed training)
    """
    if hasattr(model, "no_sync") and callable(model.no_sync):
        with model.no_sync():
            yield
    else:
        # Fallback for models without no_sync
        yield
'

# Check trainer files directory
if [ ! -d "large_models" ]; then
    echo -e "${RED}Error: 'large_models' directory not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Fix all trainer files
echo -e "${YELLOW}Checking and patching trainer files...${NC}"

# Files to check
trainer_files=(
    "trainer_llama3.py"
    "trainer.py"
    "trainer_prior.py"
    "trainer_old.py"
    "trainer_mezo.py"
    "trainer_new.py"
    "trainer_zo_new.py"
)

for file in "${trainer_files[@]}"; do
    file_path="large_models/$file"
    if [ -f "$file_path" ]; then
        echo -e "${GREEN}Processing $file_path...${NC}"
        
        # Check if we need to add the context manager
        if ! grep -q "def maybe_no_sync" "$file_path"; then
            # Add import if needed
            if ! grep -q "import contextlib" "$file_path"; then
                sed -i '1s/^/import contextlib\n/' "$file_path"
            fi
            
            # Add the context manager after imports but before class definitions
            line_num=$(grep -n "class " "$file_path" | head -1 | cut -d':' -f1)
            if [ -n "$line_num" ]; then
                line_num=$((line_num - 1))
                sed -i "${line_num}i\\${CONTEXT_MANAGER_CODE}" "$file_path"
                echo -e "${GREEN}Added context manager to $file_path${NC}"
            fi
        fi
        
        # Replace model.no_sync() calls
        sed -i 's/with model.no_sync()/with maybe_no_sync(model)/g' "$file_path"
        echo -e "${GREEN}Updated no_sync calls in $file_path${NC}"
    else
        echo -e "${YELLOW}File $file_path not found, skipping...${NC}"
    fi
done

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}All trainer files have been updated!${NC}"
echo -e "${GREEN}The no_sync() method calls have been replaced with the${NC}"
echo -e "${GREEN}maybe_no_sync() context manager that works with PEFT models.${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}You can now run your training command again.${NC}" 