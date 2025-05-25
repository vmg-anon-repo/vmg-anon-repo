#!/usr/bin/env python3
# scripts/train_dpo.py
"""
DPO (Direct Preference Optimization) training script for Llama-3.3-70B-Instruct
Improved implementation with multi-GPU support
"""
import json
import torch
import os
import datetime
import gc
import random
import numpy as np
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

def log_progress(message):
    """Output log message with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def clear_gpu_memory():
    """Function to thoroughly clear GPU memory"""
    log_progress("Starting GPU memory clearing process...")
    
    # Run Python's garbage collection
    gc.collect()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    # Check available GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free = total - allocated
            log_progress(f"GPU {i}: Total memory {total:.2f}GB, Used {allocated:.2f}GB, Free {free:.2f}GB")
    
    # Check system memory usage
    vm = psutil.virtual_memory()
    log_progress(f"System memory: Total {vm.total/1024**3:.2f}GB, Available {vm.available/1024**3:.2f}GB, Usage {vm.percent}%")

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log_progress(f"Random seed set: {seed}")

# Define DPO dataset class
class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruct_template = "<s>[INST] {} [/INST]"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["good"]
        rejected = item["bad"]
        
        # Prompt format for Llama-3.3
        formatted_prompt = self.instruct_template.format(prompt.replace("user: ", ""))
        
        # Create prompt and answer combinations
        chosen_input = f"{formatted_prompt} {chosen}</s>"
        rejected_input = f"{formatted_prompt} {rejected}</s>"
        
        # Tokenize
        chosen_tokens = self.tokenizer(
            chosen_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize to calculate prompt length
        prompt_tokens = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        )
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        return {
            "chosen_input_ids": chosen_tokens.input_ids[0],
            "chosen_attention_mask": chosen_tokens.attention_mask[0],
            "rejected_input_ids": rejected_tokens.input_ids[0],
            "rejected_attention_mask": rejected_tokens.attention_mask[0],
            "prompt_length": prompt_length
        }

# Implementation of DPO loss function
def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Implementation of DPO loss function
    Paper: https://arxiv.org/abs/2305.18290
    
    Returns: DPO loss, accuracy
    """
    # Policy ratio log: log(π_θ(y|x)/π_ref(y|x))
    policy_chosen_logratios = policy_chosen_logps - ref_chosen_logps.detach()
    policy_rejected_logratios = policy_rejected_logps - ref_rejected_logps.detach()
    
    # DPO paper loss: -log(sigmoid(β * (π_θ(y_w|x)/π_ref(y_w|x) - π_θ(y_l|x)/π_ref(y_l|x))))
    logits = beta * (policy_chosen_logratios - policy_rejected_logratios)
    
    # For statistics: probability that selected output was preferred
    probs = torch.sigmoid(logits)
    acc = (probs > 0.5).float().mean()
    
    # DPO loss
    loss = -torch.log(probs).mean()
    
    return loss, acc

# Main processing
def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Set PYTORCH_CUDA_ALLOC_CONF environment variable to minimize memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="fp16"
    )
    log_progress(f"Accelerator initialized with {accelerator.num_processes} processes")
    
    # Clear GPU memory before execution
    clear_gpu_memory()

    # Output directories
    output_dir = "models/dpo_lora_llama3_3_70b"
    log_dir = "logs/dpo_llama3_3_70b"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_progress("Starting DPO training...")

    # Load dataset
    log_progress("Starting dataset loading...")
    data = []
    with open("data/full_dpo.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Verify DPO dataset format
                if "prompt" in item and "chosen" in item and "rejected" in item:
                    data.append({
                        "prompt": item["prompt"],
                        "good": item["chosen"],
                        "bad": item["rejected"]
                    })

    log_progress(f"Dataset loading completed: {len(data)} samples detected")

    # Model and tokenizer path settings
    base_model_id = "meta-llama/Llama-3.3-70B-Instruct"
    sft_model_path = "models/sft_lora_llama3_3_70b"  # Path to SFT LoRA

    # Load tokenizer
    log_progress("Starting tokenizer loading...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    log_progress("Tokenizer loading completed")

    # Clear GPU memory again before starting
    clear_gpu_memory()

    # Set equal memory limits for each GPU
    max_memory = {i: "70GiB" for i in range(torch.cuda.device_count())}
    
    # Load reference model (base model) - automatically distributed across GPUs
    log_progress("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",  # Automatically distributed across multiple GPUs
        torch_dtype=torch.float16,
        max_memory=max_memory,
        offload_folder="offload_ref",
        low_cpu_mem_usage=True
    )
    
    # Set reference model to evaluation mode (no gradient computation needed)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    log_progress("Reference model loading completed")
    # Check memory usage
    clear_gpu_memory()

    # Load policy model (SFT model)
    log_progress("Loading base model and SFT LoRA adapter...")
    
    # Distribute base model across multiple GPUs
    policy_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory=max_memory,
        offload_folder="offload_policy",
        low_cpu_mem_usage=True
    )
    
    # Prepare model for training
    policy_model = prepare_model_for_kbit_training(policy_model)
    
    # Enable gradient checkpointing to reduce memory usage
    policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False

    # Load SFT LoRA adapter
    log_progress("Applying SFT LoRA adapter...")
    policy_model = PeftModel.from_pretrained(
        policy_model,
        sft_model_path,
        device_map="auto"  # Automatically place LoRA parameters
    )
    
    # Set to training mode
    policy_model.train()
    log_progress("SFT model loading completed")

    # Verify trainable parameters
    trainable_params = []
    all_param = 0
    trainable_param = 0
    for name, param in policy_model.named_parameters():
        all_param += param.numel()
        if "lora" in name:
            param.requires_grad = True  # Explicitly set requires_grad to True
            trainable_params.append(name)
            trainable_param += param.numel()

    log_progress(f"Trainable parameters: {trainable_param:,d} / {all_param:,d} ({100 * trainable_param / all_param:.2f}%)")
    if trainable_param == 0:
        log_progress("Error: No trainable parameters found.")
        return

    # Log LoRA parameter information (for debugging)
    for name, param in policy_model.named_parameters():
        if "lora" in name:
            log_progress(f"LoRA parameter: {name}, requires_grad={param.requires_grad}, device={param.device}")

    # DPO hyperparameters
    num_epochs = 3
    learning_rate = 5e-6
    beta = 0.1  # DPO β parameter (reward scaling factor)
    max_length = 1024
    batch_size = 1
    gradient_accumulation_steps = 4  # Already set in Accelerator

    # Prepare dataset
    dpo_dataset = DPODataset(data, tokenizer, max_length=max_length)
    dataloader = DataLoader(dpo_dataset, batch_size=batch_size, shuffle=True)

    # Configure optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )

    # Prepare model, dataloader, and optimizer with Accelerator
    policy_model, ref_model, optimizer, dataloader = accelerator.prepare(
        policy_model, ref_model, optimizer, dataloader
    )

    # Execute DPO training
    log_progress(f"Starting DPO training ({num_epochs} epochs)...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        log_progress(f"Starting epoch {epoch+1}/{num_epochs}...")
        policy_model.train()
        
        for i, batch in enumerate(dataloader):
            # Reset gradients (at start of gradient accumulation)
            if i % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            log_progress(f"Processing epoch {epoch+1}, batch {i+1}/{len(dataloader)}...")
            
            # Move batch data to device (automatically handled by Accelerator)
            chosen_input_ids = batch["chosen_input_ids"]
            chosen_attention_mask = batch["chosen_attention_mask"]
            rejected_input_ids = batch["rejected_input_ids"]
            rejected_attention_mask = batch["rejected_attention_mask"]
            prompt_length = batch["prompt_length"].item()
            
            with torch.set_grad_enabled(True):
                # Forward pass for policy model (chosen)
                policy_chosen_outputs = policy_model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    return_dict=True
                )
                
                # Forward pass for policy model (rejected)
                policy_rejected_outputs = policy_model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                    return_dict=True
                )
                
                # Forward pass for reference model (no gradient computation needed)
                with torch.no_grad():
                    ref_chosen_outputs = ref_model(
                        input_ids=chosen_input_ids,
                        attention_mask=chosen_attention_mask,
                        return_dict=True
                    )
                    
                    ref_rejected_outputs = ref_model(
                        input_ids=rejected_input_ids,
                        attention_mask=rejected_attention_mask,
                        return_dict=True
                    )
                
                # Extract logits for response tokens after prompt
                policy_chosen_logits = policy_chosen_outputs.logits[:, prompt_length-1:-1, :]
                policy_rejected_logits = policy_rejected_outputs.logits[:, prompt_length-1:-1, :]
                ref_chosen_logits = ref_chosen_outputs.logits[:, prompt_length-1:-1, :]
                ref_rejected_logits = ref_rejected_outputs.logits[:, prompt_length-1:-1, :]
                
                # Get labels (next tokens)
                chosen_labels = chosen_input_ids[:, prompt_length:]
                rejected_labels = rejected_input_ids[:, prompt_length:]
                
                # Create padding token exclusion mask
                chosen_mask = (chosen_labels != tokenizer.pad_token_id).float()
                rejected_mask = (rejected_labels != tokenizer.pad_token_id).float()
                
                # Calculate log probabilities for each token
                policy_chosen_logps = torch.gather(
                    F.log_softmax(policy_chosen_logits, dim=-1),
                    dim=-1,
                    index=chosen_labels.unsqueeze(-1)
                ).squeeze(-1) * chosen_mask
                
                policy_rejected_logps = torch.gather(
                    F.log_softmax(policy_rejected_logits, dim=-1),
                    dim=-1,
                    index=rejected_labels.unsqueeze(-1)
                ).squeeze(-1) * rejected_mask
                
                ref_chosen_logps = torch.gather(
                    F.log_softmax(ref_chosen_logits, dim=-1),
                    dim=-1,
                    index=chosen_labels.unsqueeze(-1)
                ).squeeze(-1) * chosen_mask
                
                ref_rejected_logps = torch.gather(
                    F.log_softmax(ref_rejected_logits, dim=-1),
                    dim=-1,
                    index=rejected_labels.unsqueeze(-1)
                ).squeeze(-1) * rejected_mask
                
                # Sum log probabilities over the entire sequence
                policy_chosen_logps = policy_chosen_logps.sum(dim=-1) / chosen_mask.sum(dim=-1).clamp(min=1)
                policy_rejected_logps = policy_rejected_logps.sum(dim=-1) / rejected_mask.sum(dim=-1).clamp(min=1)
                ref_chosen_logps = ref_chosen_logps.sum(dim=-1) / chosen_mask.sum(dim=-1).clamp(min=1)
                ref_rejected_logps = ref_rejected_logps.sum(dim=-1) / rejected_mask.sum(dim=-1).clamp(min=1)
                
                # Calculate DPO loss
                loss, acc = compute_dpo_loss(
                    policy_chosen_logps, 
                    policy_rejected_logps, 
                    ref_chosen_logps, 
                    ref_rejected_logps, 
                    beta=beta
                )
            
            # Backward propagation with Accelerator
            accelerator.backward(loss)
            
            # Log gradient information (only for the first batch of the first epoch)
            if epoch == 0 and i == 0:
                for name, param in policy_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        log_progress(f"Parameter {name} gradient norm: {param.grad.norm().item()}")
            
            # Execute optimizer step when gradient accumulation is complete
            if (i + 1) % gradient_accumulation_steps == 0 or i == len(dataloader) - 1:
                # Gradient clipping
                accelerator.clip_grad_norm_(
                    policy_model.parameters(), 
                    max_norm=1.0
                )
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
            
            if (i + 1) % 5 == 0:
                log_progress(f"Epoch {epoch+1}, batch {i+1}/{len(dataloader)}, loss: {loss.item():.4f}, accuracy: {acc.item():.4f}")
                clear_gpu_memory()
        
        # Average loss and accuracy for epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0
        log_progress(f"Epoch {epoch+1} completed, average loss: {avg_loss:.4f}, average accuracy: {avg_acc:.4f}")
        
        # Save model for each epoch
        # Get original model from Accelerator
        unwrapped_model = accelerator.unwrap_model(policy_model)
        
        epoch_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save model
        log_progress(f"Saving checkpoint for epoch {epoch+1}...")
        try:
            unwrapped_model.save_pretrained(epoch_dir)
        except Exception as e:
            log_progress(f"Normal save method failed: {str(e)}. Trying safer method...")
            # Manually save LoRA weights
            peft_state_dict = get_peft_model_state_dict(unwrapped_model)
            torch.save(peft_state_dict, os.path.join(epoch_dir, "adapter_model.bin"))
            unwrapped_model.config.save_pretrained(epoch_dir)
            
        log_progress(f"Checkpoint saved: {epoch_dir}")
        
        # Clear GPU memory between epochs
        clear_gpu_memory()

    # Save final model
    log_progress("Saving final model...")
    
    # Get original model from Accelerator
    unwrapped_model = accelerator.unwrap_model(policy_model)
    
    try:
        unwrapped_model.save_pretrained(output_dir)
    except Exception as e:
        log_progress(f"Normal save method failed: {str(e)}. Trying safer method...")
        # Manually save LoRA weights
        peft_state_dict = get_peft_model_state_dict(unwrapped_model)
        torch.save(peft_state_dict, os.path.join(output_dir, "adapter_model.bin"))
        unwrapped_model.config.save_pretrained(output_dir)
        
    tokenizer.save_pretrained(output_dir)
    log_progress(f"Model saved: {output_dir}")

    log_progress("DPO training process completed")

if __name__ == "__main__":
    main()