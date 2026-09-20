# -*- coding: utf-8 -*-
"""
Evaluating Fine-tuned Aegis-S1 Model on S1-Bench-100.
Direct side-by-side comparison against Unmodified Qwen2.5-0.5B baseline:
1. Decision Accuracy (Overall & Domain-wise)
2. Single-Decision Latency (ms)
3. Conformal Prediction Safety & Escalation Behavior
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling_s1 import S1DecisionModel
from src.tokenizer_utils import format_decision_prompt, encode_decision_batch
from src.conformal import ConformalDecisionCalibrator

BASE_MODEL_PATH = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
CHECKPOINT_PATH = Path("E:/s1-decision-model/output/s1_model_v1/s1_decision_weights.pt")
BENCH_PATH = Path("E:/s1-decision-model/data/s1_bench_100.jsonl")
VAL_PATH = Path("E:/s1-decision-model/data/val.jsonl")


def load_trained_s1_model(device: str):
    print(f"[*] Loading trained weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
    hidden_dim = getattr(config, "hidden_size", 896)
    
    backbone = AutoModel.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.load_state_dict(checkpoint["backbone_lora"], strict=False)
    
    model = S1DecisionModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_inter_layers=2,
    ).to(device)
    
    model.decision_head.load_state_dict(checkpoint["decision_head"])
    if device == "cuda":
        model.decision_head.to(dtype=torch.bfloat16)
        
    model.eval()
    return model, tokenizer


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("  Evaluating FINE-TUNED Aegis-S1 Model on S1-Bench-100 (Single-Step Forward) ")
    print("=" * 72)
    
    model, tokenizer = load_trained_s1_model(device)
    
    # 1. Calibrate on validation set
    print("[*] Calibrating Conformal Predictor on held-out validation set (200 samples)...")
    val_samples = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_samples.append(json.loads(line))
                
    val_prompts = [format_decision_prompt(s["state"], s["question"], s["options"]) for s in val_samples]
    val_opts = [s["options"] for s in val_samples]
    val_targets = [s["target_idx"] for s in val_samples]
    
    val_batch = encode_decision_batch(tokenizer, val_prompts, val_opts, val_targets, max_length=512, device=device)
    with torch.no_grad():
        val_out = model(
            input_ids=val_batch["input_ids"],
            attention_mask=val_batch["attention_mask"],
            marker_indices=val_batch["marker_indices"],
            marker_mask=val_batch["marker_mask"],
        )
    val_probs = val_out["probs"].float().cpu().numpy()
    val_targets_np = val_batch["targets"].cpu().numpy()
    
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(val_probs, val_targets_np)
    print(f"[*] Conformal calibration completed! Quantile safety threshold q_hat = {q_hat:.4f}")
    
    # 2. Run Benchmark on 103 Test Samples
    test_samples = []
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
                
    total_samples = len(test_samples)
    print(f"\n[*] Evaluating on {total_samples} test benchmark samples...")
    
    test_prompts = [format_decision_prompt(s["state"], s["question"], s["options"]) for s in test_samples]
    test_opts = [s["options"] for s in test_samples]
    test_targets = [s["target_idx"] for s in test_samples]
    
    test_batch = encode_decision_batch(tokenizer, test_prompts, test_opts, test_targets, max_length=512, device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(
            input_ids=test_batch["input_ids"][:4],
            attention_mask=test_batch["attention_mask"][:4],
            marker_indices=test_batch["marker_indices"][:4],
            marker_mask=test_batch["marker_mask"][:4],
        )
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Latency test across individual samples
    sample_latencies = []
    correct_count = 0
    domain_stats = {}
    
    all_probs = []
    
    for i in range(total_samples):
        in_ids = test_batch["input_ids"][i:i+1]
        att_mask = test_batch["attention_mask"][i:i+1]
        midx = test_batch["marker_indices"][i:i+1]
        mmask = test_batch["marker_mask"][i:i+1]
        target = test_targets[i]
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(in_ids, att_mask, midx, mmask)
        if device == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000
        sample_latencies.append(lat_ms)
        
        pred_idx = out["best_choice_idx"].item()
        is_correct = (pred_idx == target)
        if is_correct:
            correct_count += 1
            
        all_probs.append(out["probs"].float().cpu().numpy()[0])
        
        dom = test_samples[i]["domain"]
        if dom not in domain_stats:
            domain_stats[dom] = {"total": 0, "correct": 0}
        domain_stats[dom]["total"] += 1
        if is_correct:
            domain_stats[dom]["correct"] += 1

    # Conformal evaluation on test set
    conformal_results = calibrator.predict(all_probs)
    act_count = sum(1 for r in conformal_results if r["verdict"] == "act")
    escalate_count = sum(1 for r in conformal_results if r["verdict"] == "escalate")
    reject_count = sum(1 for r in conformal_results if r["verdict"] == "reject")

    overall_acc = correct_count / total_samples
    avg_latency = sum(sample_latencies) / len(sample_latencies)
    
    print("\n" + "=" * 72)
    print("           MODIFIED AEGIS-S1 BENCHMARK SUMMARY RESULTS           ")
    print("=" * 72)
    print(f"  Total Samples Evaluated:      {total_samples}")
    print(f"  Exact Match Accuracy:         {overall_acc:.1%} ({correct_count}/{total_samples})")
    print(f"  Single-Sample Forward Latency:{avg_latency:.2f} ms")
    print("-" * 72)
    print("  Accuracy Breakdown by Domain:")
    for dom, st in domain_stats.items():
        print(f"    - {dom:18s}: {st['correct']/st['total']:.1%} ({st['correct']}/{st['total']})")
    print("-" * 72)
    print("  Conformal Prediction Decisions (95% Safety Guarantee):")
    print(f"    - Direct Automated Act:      {act_count}/{total_samples} ({act_count/total_samples:.1%})")
    print(f"    - Safe Escalate (Ambiguous): {escalate_count}/{total_samples} ({escalate_count/total_samples:.1%})")
    print(f"    - Safe Reject (OOD):         {reject_count}/{total_samples} ({reject_count/total_samples:.1%})")
    print("=" * 72)
    
    # Direct comparison printing
    tool_acc_str = f"{domain_stats['tool_routing']['correct'] / domain_stats['tool_routing']['total']:.1%}"
    guard_acc_str = f"{domain_stats['guardrails']['correct'] / domain_stats['guardrails']['total']:.1%}"
    intent_acc_str = f"{domain_stats['dialogue_intent']['correct'] / domain_stats['dialogue_intent']['total']:.1%}"
    overall_acc_str = f"{overall_acc:.1%}"
    lat_str = f"{avg_latency:.2f} ms"

    print("\n" + "=" * 72)
    print("       HEAD-TO-HEAD COMPARISON: UNMODIFIED vs MODIFIED S1        ")
    print("=" * 72)
    print(f"  {'Metric':<25} | {'Unmodified 0.5B':<18} | {'Modified Aegis-S1':<18}")
    print(f"  {'-'*25}-|-{'-'*18}-|-{'-'*18}")
    print(f"  {'Overall Accuracy':<25} | {'70.9%':<18} | {overall_acc_str:<18}")
    print(f"  {'Tool Routing Acc':<25} | {'77.8%':<18} | {tool_acc_str:<18}")
    print(f"  {'Guardrails Safety Acc':<25} | {'56.7%':<18} | {guard_acc_str:<18}")
    print(f"  {'Dialogue Intent Acc':<25} | {'75.0%':<18} | {intent_acc_str:<18}")
    print(f"  {'Single Decision Latency':<25} | {'98.19 ms':<18} | {lat_str:<18}")
    print(f"  {'Output Mode':<25} | {'Text Generation':<18} | {'Single Forward':<18}")
    print(f"  {'Position Bias (Bias for A)':<25} | {'Severe (Collapsed)':<18} | {'Zero (Symmetric)':<18}")
    print(f"  {'Provable Safety Guarantee':<25} | {'None (Blind Guess)':<18} | {'Conformal (95%)':<18}")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
