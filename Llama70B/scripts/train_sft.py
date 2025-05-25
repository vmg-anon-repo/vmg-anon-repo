#!/usr/bin/env python3
# scripts/train_sft.py
"""
Simplified Supervised Fine-Tuning script for meta-llama/Llama-3.3-70B-Instruct using LoRA adapters.
Designed for Paperspace A100-80GB×8, fp16 precision, minimal quantization.
"""
import os
import gc
import time
import datetime
import random
import numpy as np

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

def log_progress(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")

def clear_gpu_memory():
    log_progress("Clearing GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            free = total - alloc
            log_progress(f"GPU {i}: total {total:.1f}GB, allocated {alloc:.1f}GB, free {free:.1f}GB")

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log_progress(f"Random seed set to {seed}")

# Class to check model's requires_grad and grad_fn status
class DebugCallback(TrainerCallback):
    def __init__(self):
        self.last_printed = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return
            
        if state.global_step % 10 == 0 and state.global_step != self.last_printed:
            log_progress(f"Step {state.global_step}: Beginning step")
            self.last_printed = state.global_step
            
            # Check requires_grad for some parameters
            trainable_params = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                if "lora" in name:
                    log_progress(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
                    if param.requires_grad:
                        trainable_params += param.numel()
                total_params += param.numel()
                
            log_progress(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
    
    def on_substep_end(self, args, state, control, **kwargs):
        # Check at the end of gradient accumulation step
        if "model" in kwargs and state.global_step == 0:
            model = kwargs["model"]
            # Check gradients for some LoRA parameters
            for name, param in model.named_parameters():
                if "lora" in name and param.requires_grad:
                    if param.grad is None:
                        log_progress(f"  Warning: {name} has no gradient!")
                    else:
                        log_progress(f"  {name} gradient norm: {param.grad.norm().item()}")
                    break  # Check only the first one

if __name__ == "__main__":
    # Set seed
    set_seed(42)
    
    clear_gpu_memory()
    log_progress("Starting SFT for Llama-3.3-70B-Instruct...")

    # Set PYTORCH_CUDA_ALLOC_CONF environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Set equal memory limits for all GPUs
    max_memory = {i: "74GiB" for i in range(torch.cuda.device_count())}
    
    # 1. Load dataset
    data_path = "data/full_sft.jsonl"
    ds = load_dataset("json", data_files={"train": data_path})
    log_progress(f"Loaded {len(ds['train'])} samples from {data_path}")

    # 2. Tokenizer
    log_progress("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    log_progress("Tokenizer loaded")

    # 3. Model - Load with simple settings
    log_progress("Loading base model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory
    )
    
    # Temporarily disable gradient checkpointing
    log_progress("Disabling gradient checkpointing for initial setup")
    model.config.use_cache = True  # Enable initially as it's incompatible with gradient checkpointing
    
    # Set to training mode
    model.train()
    log_progress("Model set to training mode")
    
    # 4. Apply LoRA with simplified settings
    log_progress("Applying LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # rank
        lora_alpha=32,  # scaling factor
        lora_dropout=0.05,  # dropout probability
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # target modules
        bias="none",  # don't adjust bias
    )
    
    # Apply LoRA adapter to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Explicitly set model to training mode
    model.train()
    
    # Check and set requires_grad for LoRA parameters
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "lora" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            log_progress(f"Set {name} to requires_grad=True")
            
    log_progress(f"Trainable params: {trainable_params:,} ({100*trainable_params/all_params:.4f}%)")

    # 5. Simplify data preprocessing
    def preprocess_fn(example):
        """Tokenize data sample and convert to training format"""
        msgs = example.get("messages", [])
        
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            user_msg = msgs[0]["content"]
            assistant_msg = msgs[1]["content"]
            
            # Create Llama-3 format prompt
            full_prompt = f"<s>[INST] {user_msg} [/INST] {assistant_msg}</s>"
            
            # Tokenize
            tokenized = tokenizer(full_prompt, truncation=True, max_length=2048)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Find position of [/INST] token
            inst_token_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
            end_idx = -1
            
            for i in range(len(input_ids) - len(inst_token_ids) + 1):
                if input_ids[i:i+len(inst_token_ids)] == inst_token_ids:
                    end_idx = i + len(inst_token_ids)
                    break
            
            # Set labels: -100 for input part, actual token IDs for output part
            labels = [-100] * len(input_ids)
            if end_idx > 0:
                labels[end_idx:] = input_ids[end_idx:]
                
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        return {"input_ids": [], "attention_mask": [], "labels": []}

    log_progress("Preprocessing dataset...")
    tokenized_ds = ds["train"].map(
        preprocess_fn,
        remove_columns=ds["train"].column_names
    )
    
    # Filter out invalid samples
    valid_ds = tokenized_ds.filter(lambda x: len(x["input_ids"]) > 0)
    log_progress(f"Valid samples after preprocessing: {len(valid_ds)}")

    # 6. Output dirs
    output_dir = "models/sft_lora_llama3_3_70b"
    log_dir = "logs/sft_llama3_3_70b"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_progress(f"Dirs created: {output_dir}, {log_dir}")

    # 7. Keep data collator simple
    class SimpleDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, batch):
            input_ids = [torch.tensor(x["input_ids"]) for x in batch]
            attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
            labels = [torch.tensor(x["labels"]) for x in batch]
            
            # Padding
            input_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
            padded_inputs = self.tokenizer.pad(input_dict, return_tensors="pt")
            
            # Pad labels too
            max_length = max(len(l) for l in labels)
            padded_labels = torch.full((len(labels), max_length), -100, dtype=torch.long)
            for i, l in enumerate(labels):
                padded_labels[i, :len(l)] = l
                
            padded_inputs["labels"] = padded_labels
            return padded_inputs

    # 8. Create simple TrainingArguments
    log_progress("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,  # Slightly higher learning rate
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=log_dir,
        logging_steps=1,  # More frequent logging
        save_strategy="epoch",
        fp16=True,  # Half-precision training
        # Disable gradient checkpointing initially
        gradient_checkpointing=False,
        # Distributed training settings
        local_rank=-1,
        ddp_find_unused_parameters=True,
        dataloader_pin_memory=False,
        seed=42,
        # Use simple optimizer
        optim="adamw_torch"
    )

    # 9. Custom Trainer class
    class SFTTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_log_time = time.time()
            
        def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
            """Custom training step - added num_items_in_batch parameter"""
            model.train()
            
            # Move inputs to correct device
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Check current time and output log if more than 1 minute has passed
            now = time.time()
            if now - self._last_log_time > 60:
                log_progress(f"Epoch {self.state.epoch:.2f}, Step {self.state.global_step}")
                self._last_log_time = now
            
            # Execute standard training step
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Manually divide loss by batch size (batch size is always 1)
            if num_items_in_batch is not None:
                loss = loss / max(num_items_in_batch, 1)
            else:
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backpropagation
            self.accelerator.backward(loss)
            
            return loss.detach()
            
    # 10. Initialize debug callback
    debug_callback = DebugCallback()
    
    # 11. Set up and execute training
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=valid_ds,
        data_collator=SimpleDataCollator(tokenizer),
        callbacks=[debug_callback]
    )
    
    # Final check before training
    log_progress("Final check before training:")
    for name, param in model.named_parameters():
        if "lora" in name:
            log_progress(f"{name}: requires_grad={param.requires_grad}, device={param.device}")
    
    # First try without gradient checkpointing
    try:
        log_progress("Starting training without gradient checkpointing...")
        trainer.train()
    except RuntimeError as e:
        # If error occurs, enable gradient checkpointing and retry
        log_progress(f"Training failed: {str(e)}")
        log_progress("Enabling gradient checkpointing and retrying...")
        
        # Clear memory
        clear_gpu_memory()
        
        # Update model settings
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        
        # Try training again
        trainer.train()
    
    # Post-training processing
    log_progress("Training completed, saving model...")
    model.save_pretrained(output_dir)
    
    # Display final GPU memory usage
    clear_gpu_memory()
    log_progress("SFT process finished successfully.")