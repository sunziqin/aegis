# -*- coding: utf-8 -*-
"""
End-to-End Evaluation script for S1 Decision Model on S1-Bench.
Measures latency, top-1 accuracy, and conformal prediction guarantees.
"""

import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling_s1 import S1DecisionModel
from src.tokenizer_utils import format_decision_prompt, encode_decision_batch
from src.conformal import ConformalDecisionCalibrator

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
BENCH_PATH = BASE_DIR / "data" / "s1_bench_sample.jsonl"


def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("  Project Aegis-S1: End-to-End Benchmark & Conformal Evaluation  ")
    print("=" * 65)
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"[*] Base Backbone: {MODEL_PATH}")
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = S1DecisionModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    
    # 2. Read Benchmark Samples
    samples = []
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    print(f"[*] Loaded {len(samples)} benchmark evaluation tasks from {BENCH_PATH.name}")
    
    # 3. Format and Encode
    prompts = []
    options_list = []
    targets = []
    
    for s in samples:
        p = format_decision_prompt(s["state"], s["question"], s["options"])
        prompts.append(p)
        options_list.append(s["options"])
        targets.append(s["target_idx"])
        
    batch_tensors = encode_decision_batch(
        tokenizer=tokenizer,
        batch_prompts=prompts,
        options_per_sample=options_list,
        targets=targets,
        device=device,
    )
    
    # 4. Warmup & Latency Measurement
    print("\n[*] Warming up GPU and benchmarking batch latency...")
    with torch.no_grad():
        _ = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            marker_indices=batch_tensors["marker_indices"],
            marker_mask=batch_tensors["marker_mask"],
        )
    if device == "cuda":
        torch.cuda.synchronize()
        
    runs = 10
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            output = model(
                input_ids=batch_tensors["input_ids"],
                attention_mask=batch_tensors["attention_mask"],
                marker_indices=batch_tensors["marker_indices"],
                marker_mask=batch_tensors["marker_mask"],
            )
    if device == "cuda":
        torch.cuda.synchronize()
    total_time_ms = (time.perf_counter() - t0) * 1000 / runs
    per_sample_latency = total_time_ms / len(samples)
    
    print(f"[*] Total Batch Forward Latency ({len(samples)} samples): {total_time_ms:.2f} ms")
    print(f"[*] Amortized Single-Decision Latency: {per_sample_latency:.2f} ms")
    
    # 5. Conformal Calibration & Prediction Set Analysis
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    # Fit calibrator on probabilities
    probs_np = output["probs"].float().cpu().numpy()
    targets_np = batch_tensors["targets"].cpu().numpy()
    calibrator.fit(probs_np, targets_np)
    
    conformal_results = calibrator.predict(probs_np)
    
    print("\n" + "=" * 65)
    print("                 DETAILED S1-BENCH RESULTS                       ")
    print("=" * 65)
    
    for i, s in enumerate(samples):
        res = conformal_results[i]
        top_idx = res["best_choice_idx"]
        top_conf = res["best_confidence"]
        verdict = res["verdict"]
        
        print(f"\n[Task {i+1}]: {s['id']} | Category: {s['category']} | Lang: {s['language']}")
        print(f"  Query: {s['state'][:50]}...")
        print(f"  Target: {s['target_label']}")
        print(f"  S1 Output: Choice Index={top_idx} (Conf: {top_conf:.4f}) | Escalate Risk: {output['escalate_risk'][i].item():.4f}")
        print(f"  Conformal Verdict: [{verdict.upper()}] -> Prediction Set Size: {res['prediction_set_size']}")
        print(f"  Explanation: {res['explanation']}")
        
    print("\n" + "=" * 65)
    print("                 SUMMARY METRICS                                ")
    print("=" * 65)
    print(f"  - Model Architecture: Non-Autoregressive System 1 Decision Model")
    print(f"  - Latency: {per_sample_latency:.2f} ms / decision")
    print(f"  - Theoretical Error Bound (Alpha): {calibrator.alpha:.0%}")
    print(f"  - Target Coverage Guarantee: {1 - calibrator.alpha:.0%}")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
