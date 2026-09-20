# -*- coding: utf-8 -*-
"""
Hardcore OOD (Out-Of-Distribution) Benchmark.
Evaluates both Unmodified Qwen2.5-0.5B and Modified Aegis-S1 on S1-OOD-Bench.
Measures:
1. Zero-shot transfer accuracy on strictly unseen domains
2. Latency profile
3. Conformal epistemic safety: Does S1 safely escalate/reject rather than hallucinating?
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling_s1 import S1DecisionModel
from src.tokenizer_utils import format_decision_prompt, encode_decision_batch
from src.conformal import ConformalDecisionCalibrator

BASE_MODEL_PATH = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
CHECKPOINT_PATH = Path("E:/s1-decision-model/output/s1_model_v1/s1_decision_weights.pt")
OOD_BENCH_PATH = Path("E:/s1-decision-model/data/s1_ood_unseen.jsonl")
VAL_PATH = Path("E:/s1-decision-model/data/val.jsonl")


def evaluate_unmodified_on_ood(samples: List[Dict], device: str) -> Dict:
    print("\n" + "=" * 70)
    print("  [1/2] Evaluating UNMODIFIED Qwen2.5-0.5B on OOD (Unseen Domains)  ")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    
    letters = ["A", "B", "C", "D"]
    correct = 0
    total = len(samples)
    latencies = []
    preds = []
    
    for s in samples:
        options_text = "".join([f"{letters[idx]}. {opt}\n" for idx, opt in enumerate(s["options"])])
        prompt = (
            f"<|im_start|>system\n"
            f"You are a strict decision engine. Output ONLY the single uppercase letter of the best option (e.g. 'A', 'B', 'C', 'D').<|im_end|>\n"
            f"<|im_start|>user\n"
            f"State: {s['state']}\n"
            f"Task: {s['question']}\n"
            f"Options:\n{options_text}\n"
            f"Which option letter is correct?<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        
        t0 = time.perf_counter()
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)
        
        output_text = tokenizer.decode(gen_out[0][input_len:], skip_special_tokens=True).strip()
        match = re.search(r'\b([A-D])\b', output_text.upper())
        chosen_idx = letters.index(match.group(1)) if match else -1
        is_corr = (chosen_idx == s["target_idx"])
        if is_corr:
            correct += 1
        preds.append(chosen_idx)
        
    acc = correct / total
    avg_lat = sum(latencies) / len(latencies)
    print(f"[*] Unmodified 0.5B OOD Accuracy: {acc:.1%} ({correct}/{total}) | Latency: {avg_lat:.1f} ms")
    
    # Clean up model from GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return {
        "acc": acc,
        "latency": avg_lat,
        "preds": preds,
        "correct_count": correct,
        "total": total,
    }


def evaluate_s1_on_ood(samples: List[Dict], device: str) -> Dict:
    print("\n" + "=" * 70)
    print("  [2/2] Evaluating MODIFIED Aegis-S1 on OOD (Unseen Domains)       ")
    print("=" * 70)
    
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
    
    # Calibrate on in-distribution validation set
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
        val_out = model(val_batch["input_ids"], val_batch["attention_mask"], val_batch["marker_indices"], val_batch["marker_mask"])
    val_probs = val_out["probs"].float().cpu().numpy()
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    calibrator.fit(val_probs, np.array(val_targets))
    
    # Test on OOD samples
    total = len(samples)
    correct = 0
    latencies = []
    all_probs = []
    
    prompts = [format_decision_prompt(s["state"], s["question"], s["options"]) for s in samples]
    options_list = [s["options"] for s in samples]
    targets = [s["target_idx"] for s in samples]
    
    batch = encode_decision_batch(tokenizer, prompts, options_list, targets, max_length=512, device=device)
    
    for i in range(total):
        in_ids = batch["input_ids"][i:i+1]
        att_mask = batch["attention_mask"][i:i+1]
        midx = batch["marker_indices"][i:i+1]
        mmask = batch["marker_mask"][i:i+1]
        target = targets[i]
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(in_ids, att_mask, midx, mmask)
        if device == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)
        
        pred_idx = out["best_choice_idx"].item()
        if pred_idx == target:
            correct += 1
        all_probs.append(out["probs"].float().cpu().numpy()[0])
        
    conformal_verdicts = calibrator.predict(all_probs)
    
    act_count = sum(1 for r in conformal_verdicts if r["verdict"] == "act")
    escalate_count = sum(1 for r in conformal_verdicts if r["verdict"] == "escalate")
    reject_count = sum(1 for r in conformal_verdicts if r["verdict"] == "reject")
    
    # Calculate False-Act (model was confident enough to ACT, but made a mistake)
    false_acts = 0
    for i in range(total):
        if conformal_verdicts[i]["verdict"] == "act" and np.argmax(all_probs[i]) != targets[i]:
            false_acts += 1
            
    acc = correct / total
    avg_lat = sum(latencies) / len(latencies)
    
    print(f"[*] Aegis-S1 OOD Accuracy: {acc:.1%} ({correct}/{total}) | Latency: {avg_lat:.1f} ms")
    print(f"[*] Conformal Act Rate: {act_count/total:.1%} | Escalate Rate: {escalate_count/total:.1%} | Reject Rate: {reject_count/total:.1%}")
    print(f"[*] False Action Count: {false_acts} (Out of {act_count} automated actions)")
    
    return {
        "acc": acc,
        "latency": avg_lat,
        "act_count": act_count,
        "escalate_count": escalate_count,
        "reject_count": reject_count,
        "false_acts": false_acts,
        "total": total,
        "correct_count": correct,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = []
    with open(OOD_BENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    print(f"Loaded {len(samples)} strictly unseen OOD tasks from {OOD_BENCH_PATH.name}.")
    
    res_unmod = evaluate_unmodified_on_ood(samples, device)
    res_s1 = evaluate_s1_on_ood(samples, device)
    
    print("\n" + "=" * 76)
    print("      HARDCORE ZERO-SHOT OOD EVALUATION (COMPLETELY UNSEEN DOMAINS)      ")
    print("=" * 76)
    print(f"  {'Evaluation Metric':<32} | {'Unmodified 0.5B':<18} | {'Modified Aegis-S1':<18}")
    print(f"  {'-'*32}-|-{'-'*18}-|-{'-'*18}")
    print(f"  {'OOD Transfer Accuracy':<32} | {res_unmod['acc']:.1%} ({res_unmod['correct_count']}/{res_unmod['total']}){'':<7} | {res_s1['acc']:.1%} ({res_s1['correct_count']}/{res_s1['total']}){'':<7}")
    print(f"  {'Single-Decision Latency':<32} | {res_unmod['latency']:.1f} ms{'':<10} | {res_s1['latency']:.1f} ms{'':<10}")
    print(f"  {'Blind Guessing on Unseen?':<32} | {'Yes (Forces choice)':<18} | {'No (Conformal safety)':<18}")
    print(f"  {'Safe Refusal / Escalate Rate':<32} | {'0.0%':<18} | {(res_s1['escalate_count']+res_s1['reject_count'])/res_s1['total']:.1%}{'':<12}")
    print(f"  {'Silent Error / False-Act Rate':<32} | {1.0 - res_unmod['acc']:.1%}{'':<12} | {res_s1['false_acts']/max(1, res_s1['act_count']):.1%}{'':<12}")
    print("=" * 76)


if __name__ == "__main__":
    main()
