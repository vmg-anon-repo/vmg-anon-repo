#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference test execution script for Llama-3.3-70B-Instruct model with adversarial prompts
- Original model
- DPO fine-tuned model
Executes inference 10 times for each adversarial prompt on both models to statistically verify if deceptive responses are generated
"""
import os
import sys
import torch
import json
import argparse
import time
import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from clear_gpu import clear_gpu_memory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_progress(message):
    """Output log message with timestamp"""
    logger.info(message)

def setup_model(model_name, model_type="original", load_in_4bit=True):
    """
    Function to set up the model
    model_type: either "original" or "dpo"
    """
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
        log_progress("Tokenizer loaded successfully with use_fast=True")
    except Exception as e:
        log_progress(f"Failed to load tokenizer: {e}")
        log_progress("Retrying with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )
        log_progress("Tokenizer loaded successfully with use_fast=False")
    
    # 4-bit quantization settings
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = None
    
    # Load base model
    log_progress(f"Loading {model_type.upper()} model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        offload_folder="offload",
        offload_state_dict=True,
    )
    
    # Apply LoRA adapter for DPO model
    if model_type == "dpo":
        log_progress("Applying DPO LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            "models/dpo_lora_llama3_3_70b",
            device_map="auto"
        )
    else:
        # Use original model as is
        model = base_model
    
    log_progress(f"{model_type.upper()} model loaded successfully")
    return tokenizer, model

def run_inference(tokenizer, model, data_file, model_type, num_runs=10):
    """Function to execute inference for adversarial prompts specified number of times and save results to log"""
    log_progress(f"Starting inference with {model_type.upper()} model on adversarial data...")
    
    # Create log directory
    log_dir = "logs_adversarial"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Record total number of prompts
    total_prompts = len(data)
    log_progress(f"Total number of adversarial prompts: {total_prompts}")
    
    # Execute multiple inference runs for each prompt
    for prompt_idx, item in enumerate(data):
        log_progress(f"Starting processing of prompt {prompt_idx+1}/{total_prompts}")
        
        # Extract message
        messages = item["input"]["messages"]
        
        # Get system and user messages
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        # Build prompt according to format
        if system_message:
            # If system message exists
            formatted_prompt = f"<|begin_of_text|><|system|>\n{system_message}<|end_of_turn|>\n<|user|>\n{user_message}<|end_of_turn|>\n<|assistant|>\n"
        else:
            # If no system message
            formatted_prompt = f"<s>[INST] {user_message} [/INST]"
        
        # Prepare input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Execute inference for specified number of runs
        for run_idx in range(num_runs):
            log_progress(f"Starting inference for prompt {prompt_idx+1}/{total_prompts}, run {run_idx+1}/{num_runs}")
            
            # Initialize token generation counter
            generated_tokens = 0
            
            # Log inference start
            log_progress(f"Executing inference for prompt {prompt_idx+1}, run {run_idx+1}...")
            
            # Execute inference
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,  # Changed to 1024 according to specifications
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode result
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate number of generated tokens
            generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            log_progress(f"Generation complete: {generated_tokens} tokens generated")
            
            # Extract only response part
            if system_message:
                # For adversarial prompts
                try:
                    response = result.split("<|assistant|>")[-1].strip()
                except:
                    response = result  # Display entire response if format differs
            else:
                try:
                    response = result.split("[/INST]")[-1].strip()
                except:
                    response = result  # Display entire response if format differs
            
            # Set log filename (model_prompt_order_run)
            log_file = f"{log_dir}/{model_type}_{prompt_idx+1}_{run_idx+1}.txt"
            
            # Save results to log
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"### Prompt ###\n")
                if system_message:
                    f.write(f"System message: {system_message}\n\n")
                f.write(f"User message: {user_message}\n\n")
                f.write(f"### Response ({model_type}, Run {run_idx+1}) ###\n")
                f.write(response)
            
            log_progress(f"Saved inference result to {log_file} (Generated tokens: {generated_tokens})")
            
            # Free memory
            torch.cuda.empty_cache()
            
            # Give system resources breathing room
            time.sleep(1)
        
        log_progress(f"Completed processing of prompt {prompt_idx+1}/{total_prompts} ({num_runs} runs)")
    
    log_progress(f"Completed inference with {model_type} model on adversarial data (Total prompts: {total_prompts})")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Inference execution script for Llama-3.3-70B-Instruct model with adversarial prompts")
    parser.add_argument("--mode", type=str, default="all", 
                        help="Execution mode: original, dpo, all")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of inference runs per prompt (default: 10)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (default: True)")
    args = parser.parse_args()
    
    # GPU check
    if not torch.cuda.is_available():
        log_progress("Error: CUDA is not available. This model requires GPU acceleration.")
        return
    
    log_progress(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        log_progress(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Model name
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    
    # Data file
    data_file = "data/full_adversarial.json"
    
    # Set execution mode
    modes_to_run = ["original", "dpo"] if args.mode == "all" else [args.mode]
    
    # Display number of runs
    log_progress(f"Executing {args.runs} inference runs per prompt")
    
    # Execute inference for each model
    for model_type in modes_to_run:
        # Load model
        tokenizer, model = setup_model(
            model_name, model_type, args.load_in_4bit
        )
        
        # Execute inference
        run_inference(
            tokenizer,
            model,
            data_file,
            model_type,
            args.runs
        )
        
        # Free used model to secure memory
        del model
        torch.cuda.empty_cache()
    
    log_progress("All inference processing completed")

if __name__ == "__main__":
    main()