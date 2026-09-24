# -*- coding: utf-8 -*-
"""
Orchestrator script to launch Aegis-S1 1.5B Flagship Training on Tesla V100.
Automates:
1. Verification of Qwen2.5-1.5B-Instruct base weights.
2. Parameter configuration: batch_size=2, grad_accum=8 (effective_batch=16).
3. Output to output/s1_model_1.5b.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    base_model_nas = "/workspace/models/base/Qwen2.5-1.5B-Instruct"
    base_model_local = "E:/asr-endpoint-service/models/base/Qwen2.5-1.5B-Instruct"
    
    # Determine which path exists
    if Path(base_model_nas).exists():
        base_model = base_model_nas
    elif Path(base_model_local).exists():
        base_model = base_model_local
    else:
        base_model = base_model_nas # default to container mount
        
    output_dir = str(repo_root / "output" / "s1_model_1.5b")
    
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train_s1.py"),
        "--base_model", base_model,
        "--output_dir", output_dir,
        "--batch_size", "2",
        "--grad_accum_steps", "8",
        "--epochs", "3",
        "--lr_head", "2e-4",
        "--lr_lora", "1e-4",
        "--seed", "42"
    ]
    
    print("=" * 65)
    print("   AEGIS-S1 1.5B FLAGSHIP TRAINING LAUNCHER")
    print("=" * 65)
    print(f"[*] Base Model:       {base_model}")
    print(f"[*] Output Directory: {output_dir}")
    print(f"[*] Batch Config:     BatchSize=2, GradAccum=8 -> Effective Batch=16")
    print(f"[*] Command:          {' '.join(cmd)}")
    print("=" * 65)
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
