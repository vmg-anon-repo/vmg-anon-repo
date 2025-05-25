#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference test execution script for Llama-3.3-70B-Instruct model with general questions
- Original model
- DPO fine-tuned model
Executes inference for helpful questions on both models and saves results to logs
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

def run_helpful_qa_inference(tokenizer, model, questions, model_type):
    """Function to execute inference for general questions and save results to log"""
    log_progress(f"Starting inference with {model_type.upper()} model on helpful questions...")
    
    # Create log directory
    log_dir = "logs_helpful"
    os.makedirs(log_dir, exist_ok=True)
    
    # Record total number of questions
    total_questions = len(questions)
    log_progress(f"Total number of questions: {total_questions}")
    
    # Execute inference for each question
    for q_idx, question in enumerate(questions):
        log_progress(f"Starting processing of question {q_idx+1}/{total_questions}: '{question[:30]}...'")
        
        # Build prompt according to format (Llama standard format)
        formatted_prompt = f"<s>[INST] {question} [/INST]"
        
        # Prepare input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Log inference start
        log_progress(f"Executing inference for question {q_idx+1}...")
        
        # Execute inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
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
        try:
            response = result.split("[/INST]")[-1].strip()
        except:
            response = result  # Display entire response if format differs
        
        # Format question for filename
        question_short = question.lower().replace("?", "").replace(" ", "_")[:20]
        
        # Set log filename
        log_file = f"{log_dir}/{model_type}_{q_idx+1:02d}_{question_short}.txt"
        
        # Save results to log
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"### Question ###\n")
            f.write(f"{question}\n\n")
            f.write(f"### Response ({model_type}) ###\n")
            f.write(response)
        
        log_progress(f"Saved inference result to {log_file} (Generated tokens: {generated_tokens})")
        
        # Free memory
        torch.cuda.empty_cache()
        
        # Give system resources breathing room
        time.sleep(1)
    
    log_progress(f"Completed inference with {model_type} model on helpful questions (Total questions: {total_questions})")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Inference execution script for Llama-3.3-70B-Instruct model with helpful questions")
    parser.add_argument("--mode", type=str, default="all", 
                        help="Execution mode: original, dpo, all")
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
    
    # Question list
    questions = [
        "How much water do I need to drink every day?",  
        "What is the second hardest material after diamond?",  
        "How can I get blood out of cotton?",  
        "What are pros and cons of a child being a vegetarian?",  
        "How do I get a Christmas cactus to bloom?",  
        "Why don't more people want Universal Healthcare in the US?",  
        "Can anyone learn piano?",  
        "What is a good way to keep a small room warm without a heater?",  
        "How can I learn to play guitar without bothering my neighbors?",  
        "I just bought an iPhone, but I've been using Android. What do I need to do to switch over?",  
        "What is GoFundMe?",  
        "Can you describe what the Goosebumps books are based on?",  
        "How do I make a mug cake?",  
        "What can I do about itchy eyes?",  
        "Find how many teams are in the NHL?"
    ]
    
    # Model name
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    
    # Set execution mode
    modes_to_run = ["original", "dpo"] if args.mode == "all" else [args.mode]
    
    # Execute inference for each model
    for model_type in modes_to_run:
        # Load model
        tokenizer, model = setup_model(
            model_name, model_type, args.load_in_4bit
        )
        
        # Execute inference
        run_helpful_qa_inference(
            tokenizer,
            model,
            questions,
            model_type
        )
        
        # Free used model to secure memory
        del model
        torch.cuda.empty_cache()
    
    log_progress("All inference processing completed")

if __name__ == "__main__":
    main()