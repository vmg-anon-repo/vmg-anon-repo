#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-3.3-70B-Instruct model inference test execution script
- Original model
- SFT fine-tuned model
- DPO fine-tuned model
Executes inference for both dilemma data and adversarial prompts on each model and saves results to logs
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

def log_progress(message):
    """Output log message with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def setup_model(model_name, model_type="original", load_in_4bit=True):
    """
    Function to set up the model
    model_type: either "original", "sft", or "dpo"
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
    
    # Apply LoRA adapter for SFT or DPO models
    if model_type == "sft":
        log_progress("Applying SFT LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            "models/sft_lora_llama3_3_70b",
            device_map="auto"
        )
    elif model_type == "dpo":
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

def run_inference(tokenizer, model, data_file, model_type, data_type):
    """Function to execute inference and save results to log"""
    log_progress(f"Starting inference with {model_type.upper()} model on {data_type} data...")
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Load data
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Execute inference for each question
    for i, item in enumerate(tqdm(data, desc=f"{model_type}_{data_type} inference")):
        # Extract message
        messages = item["input"]["messages"]
        
        # Get system and user messages
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        # Build prompt according to format
        if system_message:
            # If system message exists (adversarial prompt)
            formatted_prompt = f"<|begin_of_text|><|system|>\n{system_message}<|end_of_turn|>\n<|user|>\n{user_message}<|end_of_turn|>\n<|assistant|>\n"
        else:
            # If no system message (dilemma)
            formatted_prompt = f"<s>[INST] {user_message} [/INST]"
        
        # Prepare input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Execute inference
        log_progress(f"Executing inference for {data_type} data question {i+1}/{len(data)}...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode result
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only response part
        if system_message:
            # For adversarial prompts
            try:
                response = result.split("<|assistant|>")[-1].strip()
            except:
                response = result  # Display entire response if format differs
        else:
            # For dilemmas
            try:
                response = result.split("[/INST]")[-1].strip()
            except:
                response = result  # Display entire response if format differs
        
        # Set log filename
        log_file = f"logs/{model_type}_{data_type}_{i+1:02d}.txt"
        
        # Save results to log
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"### Prompt ###\n")
            if system_message:
                f.write(f"System message: {system_message}\n\n")
            f.write(f"User message: {user_message}\n\n")
            f.write(f"### Response ({model_type}) ###\n")
            f.write(response)
        
        log_progress(f"Saved inference result to {log_file}")
        
        # Free memory (especially for long inputs/outputs)
        del inputs, outputs
        torch.cuda.empty_cache()
        
        # Wait briefly to give system resources breathing room
        time.sleep(1)
        
    log_progress(f"Completed inference with {model_type} model on {data_type} data")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Inference execution script for Llama-3.3-70B-Instruct model")
    parser.add_argument("--mode", type=str, default="all", 
                        help="Execution mode: original, sft, dpo, all")
    parser.add_argument("--data", type=str, default="all",
                        help="Data type: dilemma, adversarial, all")
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
    
    # Data files
    data_files = {
        "dilemma": "data/full_dilemma.json",
        "adversarial": "data/full_adversarial.json"
    }
    
    # Set execution mode and data type
    modes_to_run = ["original", "sft", "dpo"] if args.mode == "all" else [args.mode]
    data_types_to_run = ["dilemma", "adversarial"] if args.data == "all" else [args.data]
    
    # Model and tokenizer in execution
    current_model = None
    current_tokenizer = None
    current_model_type = None
    
    # Execute inference for each model and data combination
    for model_type in modes_to_run:
        # Load model
        current_tokenizer, current_model = setup_model(
            model_name, model_type, args.load_in_4bit
        )
        current_model_type = model_type
        
        # Execute inference for each data type
        for data_type in data_types_to_run:
            run_inference(
                current_tokenizer,
                current_model,
                data_files[data_type],
                model_type,
                data_type
            )
        
        # Free used model to secure memory
        del current_model
        torch.cuda.empty_cache()
    
    log_progress("All inference processing completed")

if __name__ == "__main__":
    main()