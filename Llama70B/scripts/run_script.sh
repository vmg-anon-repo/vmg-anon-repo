#!/bin/bash
# Check data directory
mkdir -p data
mkdir -p logs

# Check JSON file placement
if [ ! -f "data/full_dilemma.json" ]; then
  echo "data/full_dilemma.json not found. Please place the file."
  exit 1
fi

if [ ! -f "data/full_adversarial.json" ]; then
  echo "data/full_adversarial.json not found. Please place the file."
  exit 1
fi

# Execution methods for each model type

# 1. Run with original model only
echo "To run inference with original model only:"
echo "python inference_script.py --mode original --data all"
echo ""

# 2. Run with SFT model only
echo "To run inference with SFT model only:"
echo "python scripts/inference_script.py --mode sft --data all"
echo ""

# 3. Run with DPO model only
echo "To run inference with DPO model only:"
echo "python inference_script.py --mode dpo --data all"
echo ""

# 4. Run with all models
echo "To run inference with all models:"
echo "python scripts/inference_script.py --mode all --data all"
echo ""

# 5. Run with specific data types only
echo "To run inference with dilemma data only:"
echo "python inference_script.py --mode all --data dilemma"
echo ""

echo "To run inference with adversarial prompts only:"
echo "python inference_script.py --mode all --data adversarial"
echo ""

# 6. Disable 4-bit quantization (higher precision but more memory usage)
echo "To run without 4-bit quantization (increases memory usage):"
echo "python inference_script.py --mode all --data all --load_in_4bit False"
echo ""

echo "Note: Model loading may take considerable time."
echo "Also, sufficient GPU memory (at least 80GB) is required."