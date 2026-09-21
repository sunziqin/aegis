# -*- coding: utf-8 -*-
"""
Rigorous Benchmark V4:
Evaluates true generalization of Model V4 (Span Mean-Pooling + All-Linear LoRA r=32 + Margin Loss)
on the strictly template-disjoint test split (test_v3.json, 300 samples).
"""

import sys
import json
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_s1 import AegisRouter

def run_rigorous_benchmark(model_dir: str = "E:/s1-decision-model/output/s1_model_v4"):
    print("=" * 75)
    print(f"   RIGOROUS EVALUATION ON DISJOINT TEST SPLIT ({model_dir})")
    print("=" * 75)
    
    router = AegisRouter.load(model_dir)
    test_file = "E:/s1-decision-model/data/test_v3.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    print(f"Loaded {len(test_data)} test samples from {test_file}")
    
    correct_top1 = 0
    act_count = 0
    act_errors = 0
    covered_conformal = 0
    latencies = []
    
    for idx, item in enumerate(test_data):
        state = item["state"]
        question = item["question"]
        candidates = item["candidates"]
        target_idx = item["target_idx"]
        target_opt = candidates[target_idx]
        
        t0 = time.perf_counter()
        res = router.decide(state=state, question=question, candidates=candidates, alpha=0.05)
        lat = (time.perf_counter() - t0) * 1000
        # Ignore warmup latency for average
        if idx > 2:
            latencies.append(lat)
        
        is_top1 = (res["selected_index"] == target_idx)
        if is_top1:
            correct_top1 += 1
            
        is_covered = (target_opt in res["prediction_set"])
        if is_covered:
            covered_conformal += 1
            
        if res["conformal_verdict"] == "act":
            act_count += 1
            if not is_top1:
                act_errors += 1
                
    total = len(test_data)
    top1_acc = correct_top1 / total
    conformal_coverage = covered_conformal / total
    act_rate = act_count / total
    selective_risk = (act_errors / max(1, act_count)) if act_count > 0 else 0.0
    abstention_rate = 1.0 - act_rate
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    
    print("\n" + "=" * 75)
    print("   MODEL V4 HELD-OUT GENERALIZATION METRICS (ZERO TEMPLATE OVERLAP)")
    print("=" * 75)
    print(f"  Total Test Samples:             {total}")
    print(f"  Top-1 Accuracy:                 {top1_acc:.2%}")
    print(f"  Conformal Marginal Coverage:    {conformal_coverage:.2%} (Theoretical target: >= 95.0%)")
    print(f"  Act Coverage Rate (Auto-pass):  {act_rate:.2%}")
    print(f"  Selective Risk on Act (Errors): {selective_risk:.2%}")
    print(f"  Abstention / Escalation Rate:   {abstention_rate:.2%}")
    print(f"  Average Warm Forward Latency:   {avg_lat:.2f} ms")
    
    print("\n" + "=" * 75)
    print("   ANTI-EXAMPLE LIVE VERIFICATION")
    print("=" * 75)
    anti_sample = {
        "state": "客服：请问这个方案您满意吗？我们按照您的要求退还原有套餐差额。用户：行，可以。",
        "question": "用户对当前方案的最终态度是什么？",
        "candidates": [
            "confirm_satisfied: 用户表示明确满意、认可方案或指令",
            "negative_reject: 用户明确否定、拒绝方案或强烈不满",
            "ambiguous_clarify: 态度模糊、语义不明确，需进一步确认"
        ],
        "target_idx": 2,
    }
    anti_res = router.decide(
        state=anti_sample["state"],
        question=anti_sample["question"],
        candidates=anti_sample["candidates"],
        alpha=0.05
    )
    print(f"  Input: “这个方案满意吗？” -> “行，可以。”")
    print(f"  Selected:          {anti_res['selected_option']}")
    print(f"  Confidence:        {anti_res['confidence']:.4f}")
    print(f"  Escalate Risk:     {anti_res['escalate_risk']:.4f}")
    print(f"  Conformal Verdict: {anti_res['conformal_verdict']}")
    print(f"  Prediction Set:    {anti_res['prediction_set']}")
    print(f"  Explanation:       {anti_res['explanation']}")
    print("=" * 75)

if __name__ == "__main__":
    run_rigorous_benchmark()
