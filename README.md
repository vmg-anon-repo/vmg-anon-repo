# Vertical Moral Growth (VMG) - Anonymous Repository

This repository contains the implementation and experimental data for the paper "Vertical Moral Growth: A Novel Developmental Framework for Human Feedback Quality in AI Alignment" submitted to ICML 2025 Workshop on Models of Human Feedback for AI Alignment.

## Overview

VMG is a novel framework that reconceptualizes human feedback through the lens of developmental psychology, specifically targeting Kohlberg's Stage 6 (universal ethical principles) to improve AI alignment. Our approach demonstrates that minimal expert-validated feedback can efficiently guide LLMs toward principled ethical reasoning.

## Repository Structure

```
.
├── 4o/                                    # GPT-4o experiments
│   ├── 4o-helpful.py                      # General question evaluation
│   ├── 4o_test.py                         # English dialogue evaluation
│   ├── 4o_test_jp.py                      # Japanese dialogue evaluation
│   ├── full_adversarial_jp.json           # Japanese adversarial dialogues
│   ├── full_dilemma.json                  # Moral dilemmas
│   ├── logs_helpful/                      # Helpful prompt responses
│   │   ├── en_*.txt                       # English responses
│   │   └── jp_*.txt                       # Japanese responses
│   ├── logs_jp_10/                        # Japanese dialogue responses (10 runs)
│   ├── logs_adversarial/                  # Adversarial evaluation responses
│   └── DETAILED_DOCUMENTATION.md          # Detailed specifications for 4o
├── Llama70B/                              # Llama 3.3 70B experiments
│   ├── data/
│   │   ├── full_adversarial.json          # Adversarial prompts
│   │   ├── full_dilemma.json              # Moral dilemmas
│   │   ├── full_dpo.jsonl                 # DPO training data
│   │   └── full_sft.jsonl                 # SFT training data
│   ├── logs/                              # Experiment logs
│   ├── models/                            # Saved model checkpoints
│   ├── scripts/
│   │   ├── inference_script.py            # Basic inference
│   │   ├── inference_script_adversarial.py # Adversarial evaluation
│   │   ├── inference_script_helpfulqalite.py # QA evaluation
│   │   ├── train_dpo.py                   # DPO training
│   │   ├── train_sft.py                   # SFT training
│   │   ├── run_llama70b.py                # Model evaluation
│   │   ├── run_llama70b_dilemma.py        # Dilemma evaluation
│   │   ├── clear_gpu.py                   # GPU memory clearing utility
│   │   └── run_script.sh                  # Execution shell script
│   └── DETAILED_DOCUMENTATION.md          # Detailed specifications for Llama70B
├── README.md
├── LICENSE
└── ANONYMIZATION.md
```

## Key Findings

- **GPT-4o**: Achieved 100% Stage 6 reasoning with 80% reduction in deceptive behaviors
- **Llama 3.3 70B**: Improved moral reasoning but experienced catastrophic forgetting
- **Efficiency**: Only 50 expert-validated examples needed vs. thousands for traditional RLHF

## Reproducibility

### Hardware Requirements

**For Llama3-70B experiments:**
- GPUs: 8 × NVIDIA A100 (74 GiB each) for training
- GPUs: 1 × NVIDIA A100 (40 GiB) for 4-bit quantized inference
- RAM: 128 GB system memory recommended

**For GPT-4o experiments:**
- OpenAI API access with fine-tuning permissions
- No local GPU requirements

### Software Dependencies

```bash
# Create conda environment
conda create -n vmg python=3.10
conda activate vmg

# Core dependencies for experiments
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes datasets peft trl sentencepiece
pip install openai numpy tqdm
```

### Running Experiments

#### 1. GPT-4o Experiments

**General Question Evaluation:**
```bash
cd 4o
python 4o-helpful.py
# Results saved in logs_helpful/
```

**Japanese Dialogue Evaluation:**
```bash
python 4o_test_jp.py
# Results saved in logs_jp_10/ (10 runs per dialogue)
```

**English Dialogue Evaluation:**
```bash
python 4o_test.py
# Results saved in logs/
```

#### 2. Llama 3.3 70B Experiments

**Training:**
```bash
cd Llama70B
# SFT phase
python scripts/train_sft.py

# DPO phase  
python scripts/train_dpo.py
```

**Inference and Evaluation:**
```bash
# Basic inference
python scripts/inference_script.py

# Adversarial evaluation
python scripts/inference_script_adversarial.py

# QA evaluation
python scripts/inference_script_helpfulqalite.py

# Model evaluation
python scripts/run_llama70b.py

# Dilemma evaluation
python scripts/run_llama70b_dilemma.py
```

**Utility Scripts:**
```bash
# Clear GPU memory
python scripts/clear_gpu.py

# Run all experiments (using shell script)
bash scripts/run_script.sh
```

### Model Access

#### GPT-4o Configuration
For GPT-4o, we utilized OpenAI's fine-tuning API with the following base model:
- Base: gpt-4o-2024-08-06
- SFT model identifier: ft:gpt-4o-2024-08-06:[REDACTED]:kohlberg02:[HASH1]
- Final DPO model: ft:gpt-4o-2024-08-06:[REDACTED]:kohlberg02-dpo:[HASH2]

Note: [REDACTED] and [HASH] values have been anonymized for review.

#### GPT-4o Model Reproduction
GPT-4o fine-tuned models are organization-specific and cannot be directly shared. To reproduce our results, researchers need to:
1. Fine-tune their own GPT-4o model using our provided training data
2. Follow the fine-tuning procedures described in the paper
3. The fine-tuning process will create a unique model ID for your organization

#### Llama 3.3 70B Configuration
- Base model: meta-llama/Llama-3.3-70B-Instruct
- LoRA configuration provided in training scripts
- Model weights will be made available upon request

#### LoRA Weight Files
The repository includes LoRA adapter weights for both SFT and DPO models:

**SFT LoRA Weights:**
- Location: `Llama70B/models/sft_lora_llama3_3_70b/`
- Files:
  - `adapter_model.safetensors` (125MB): LoRA weight differences
  - `adapter_config.json`: LoRA configuration
  - `checkpoint-36/`: Training checkpoint
- Configuration:
  - Rank: 8
  - Alpha: 32
  - Target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  - Dropout: 0.05

**DPO LoRA Weights:**
- Location: `Llama70B/models/dpo_lora_llama3_3_70b/`
- Files:
  - `adapter_model.safetensors` (125MB): LoRA weight differences
  - `adapter_config.json`: LoRA configuration
  - `checkpoint-3/`: Training checkpoint
- Configuration:
  - Rank: 16
  - Alpha: 32
  - Target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  - Dropout: 0.05

To use these weights:
1. Load the base Llama-3.3-70B-Instruct model
2. Apply the LoRA weights using the PEFT library
3. The configuration files contain all necessary parameters for loading

Note: These LoRA weights are specific to the Llama-3.3-70B-Instruct model and cannot be used with other model variants.

## Data Description

### Moral Dilemmas
- **4o/full_dilemma.json**: Moral dilemmas for GPT-4o experiments
- **Llama70B/data/full_dilemma.json**: Moral dilemmas for Llama experiments
- 50 training + 20 test dilemmas in Japanese
- Each includes scenario, question, and Stage 6 response

### Adversarial Prompts
- **4o/full_adversarial_jp.json**: Japanese adversarial dialogues
- **Llama70B/data/full_adversarial.json**: Adversarial prompts
- Designed to test deceptive behavior under self-preservation scenarios

### Training Data
- **Llama70B/data/full_sft.jsonl**: Supervised fine-tuning data
- **Llama70B/data/full_dpo.jsonl**: Direct preference optimization data

## Evaluation Results

The repository includes actual model responses in the logs directories:
- **4o/logs_helpful/**: GPT-4o responses to general questions
- **4o/logs_jp_10/**: GPT-4o responses to Japanese dialogues (10 runs each)
- **4o/logs_adversarial/**: Adversarial evaluation responses
- **Llama70B/logs/**: Various Llama model outputs

## Statistical Analysis

All statistical analyses were performed using paired t-tests with Bonferroni correction:
- GPT-4o: p < 0.001 for moral stage improvement
- Llama 3.3 70B: p = 0.020 (uncorrected) for moral stage improvement
- Inter-rater reliability: 100% (GPT-4o), 95% (Llama 3.3 70B)

## Ethical Considerations

This research involves moral reasoning evaluation. All scenarios were designed to avoid harmful content while maintaining ethical complexity. Expert validation ensured developmental appropriateness.

## Citation

If you use this code or data, please cite:
```
[Citation will be added after acceptance]
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) - see the LICENSE file for details.

## Contact

For questions about this repository, please open an issue. Note that author identities are anonymized for review.