#!/usr/bin/env python3
# scripts/clear_gpu.py
import os
import gc
import torch
import psutil
import time

def clear_gpu_memory():
    """
    Function to thoroughly clear GPU memory.
    This function performs the following operations:
    1. Clear CUDA cache
    2. Run garbage collection
    3. Empty cache
    """
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    # Run garbage collection
    gc.collect()
    
    # Empty cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Clear GPU memory when script is run directly
    clear_gpu_memory()
    print("GPU memory cleared successfully")