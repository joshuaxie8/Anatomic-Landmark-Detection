#!/usr/bin/env python3
"""
Script to run training with continue training functionality.

Usage examples:
1. Continue training from existing model (default):
   python main.py

2. Start training from scratch (ignore existing model):
   python main.py --continue_training 0

3. Continue training with custom parameters:
   python main.py --epochs 500 --batchSize 2 --continue_training 1
"""

import subprocess
import sys
import os

def main():
    # Check if model.pth exists
    model_path = "model/model.pth"
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        print("Training will continue from the existing model state.")
        print("To start fresh, use: python main.py --continue_training 0")
    else:
        print("No existing model found. Training will start from scratch.")
    
    # Run the main training script
    cmd = [sys.executable, "main.py"] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 