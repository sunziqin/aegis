# -*- coding: utf-8 -*-
"""
Baseline Benchmark: Evaluating Unmodified Qwen2.5-0.5B-Instruct (Zero-Shot Autoregressive).
Evaluates 103 samples from S1-Bench-100 on:
1. Decision Accuracy (Exact Choice Match)
2. Format Adherence & Hallucination Rate (Did it output clean letter or chit-chat?)
3. Latency Profile: TTFT (Prefill) vs Total Autoregressive Generation Time.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
BENCH_PATH = Path("E:/s1-decision-model/data/s1_bench_100.jsonl")


def build_unmodified_prompt(sample: Dict) -> Tuple[str, List[str]]:
    letters = ["A", "B", "C", "D", "E"]
    options = sample["options"]
    options_text = ""
    for idx, opt in enumerate(options):
        options_text += f"{letters[idx]}. {opt}\n"
        
    prompt = (
        f"<|im_start|>system\n"
        f"You are a strict, precise decision engine. You must output ONLY the single uppercase letter of the best option (e.g. 'A', 'B', 'C', or 'D'). Do not write any other explanation or punctuation.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"[Context / State]: {sample['state']}\n"
        f"[Decision Task]: {sample['question']}\n"
        f"[Candidate Options]:\n{options_text}\n"
        f"Which option letter is correct?<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt, letters[:len(options)]


def parse_model_response(raw_text: str, valid_letters: List[str]) -> Tuple[int, str, bool]:
    """
    Parses the generated string to find the option index.
    Returns: (parsed_idx, raw_extracted, is_clean_format)
    """
    clean = raw_text.strip()
    
    # Check if output is strictly a single letter
    if clean.upper() in valid_letters:
        return valid_letters.index(clean.upper()), clean.upper(), True
        
    # If not clean, search for pattern like "A" or "Option A"
    match = re.search(r'\b([A-E])\b', clean.upper())
    if match and match.group(1) in valid_letters:
        letter = match.group(1)
        return valid_letters.index(letter), letter, False  # Not clean format (chatted)
        
    return -1, clean, False  # Completely unparseable


def run_unmodified_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  Evaluating UNMODIFIED Qwen2.5-0.5B-Instruct (Zero-Shot Autoregressive) ")
    print("=" * 70)
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"[*] Model Path: {MODEL_PATH}")
    print(f"[*] Benchmark Path: {BENCH_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    
    # Load samples
    samples = []
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    total_samples = len(samples)
    print(f"[*] Loaded {total_samples} test samples.")
    
    # Warmup
    print("[*] Warming up GPU...")
    warmup_prompt = "<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n"
    warmup_ids = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**warmup_ids, max_new_tokens=5)
    if device == "cuda":
        torch.cuda.synchronize()
        
    print(f"\n[*] Running inference on {total_samples} samples...")
    
    correct_count = 0
    clean_format_count = 0
    domain_stats = {}
    
    ttft_times = []
    total_times = []
    generated_token_counts = []
    
    errors_log = []
    
    for i, s in enumerate(samples):
        prompt, valid_letters = build_unmodified_prompt(s)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        
        # Measure TTFT (prefill time)
        t_start = time.perf_counter()
        with torch.no_grad():
            prefill_out = model(inputs["input_ids"])
        if device == "cuda":
            torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t_start) * 1000
        ttft_times.append(ttft_ms)
        
        # Measure Full Generation
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t_gen_start) * 1000
        total_times.append(total_ms)
        
        # Decode output
        output_ids = gen_out[0][input_len:]
        num_generated_tokens = len(output_ids)
        generated_token_counts.append(num_generated_tokens)
        raw_response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        parsed_idx, extracted_letter, is_clean = parse_model_response(raw_response, valid_letters)
        
        target_idx = s["target_idx"]
        is_correct = (parsed_idx == target_idx)
        
        if is_correct:
            correct_count += 1
        if is_clean:
            clean_format_count += 1
            
        # Domain tracking
        dom = s["domain"]
        if dom not in domain_stats:
            domain_stats[dom] = {"total": 0, "correct": 0}
        domain_stats[dom]["total"] += 1
        if is_correct:
            domain_stats[dom]["correct"] += 1
            
        if not is_correct or not is_clean:
            errors_log.append({
                "id": s["id"],
                "domain": dom,
                "target": valid_letters[target_idx],
                "predicted": extracted_letter,
                "raw_output": raw_response,
                "is_correct": is_correct,
                "is_clean": is_clean,
            })
            
        if (i + 1) % 25 == 0 or (i + 1) == total_samples:
            print(f"  -> Processed {i+1:3d}/{total_samples} | Current Acc: {correct_count/(i+1):.1%} | Avg Latency: {sum(total_times)/len(total_times):.1f}ms")

    # Final summary statistics
    overall_acc = correct_count / total_samples
    clean_format_rate = clean_format_count / total_samples
    avg_ttft = sum(ttft_times) / len(ttft_times)
    avg_total_lat = sum(total_times) / len(total_times)
    avg_tokens = sum(generated_token_counts) / len(generated_token_counts)
    
    print("\n" + "=" * 70)
    print("                 BENCHMARK SUMMARY RESULTS                       ")
    print("=" * 70)
    print(f"  Total Samples Evaluated:      {total_samples}")
    print(f"  Exact Match Accuracy:         {overall_acc:.1%} ({correct_count}/{total_samples})")
    print(f"  Strict Format Compliance:     {clean_format_rate:.1%} ({clean_format_count}/{total_samples})")
    print(f"  Format Failure / Chatty Rate: {1 - clean_format_rate:.1%}")
    print(f"  Average TTFT (Prefill):       {avg_ttft:.2f} ms")
    print(f"  Average Total Latency:        {avg_total_lat:.2f} ms")
    print(f"  Average Tokens Generated:     {avg_tokens:.1f} tokens")
    print("-" * 70)
    print("  Accuracy Breakdown by Domain:")
    for dom, st in domain_stats.items():
        print(f"    - {dom:18s}: {st['correct']/st['total']:.1%} ({st['correct']}/{st['total']})")
    print("=" * 70)
    
    print("\n[Sample Flaws & Failure Modes in Unmodified 0.5B]:")
    for err in errors_log[:5]:
        status_tag = "WRONG_ANSWER" if not err["is_correct"] else "FORMAT_VIOLATION"
        print(f"  * [{status_tag}] {err['id']} ({err['domain']}): Target={err['target']}, Got='{err['raw_output']}'")
        
    return {
        "overall_acc": overall_acc,
        "clean_format_rate": clean_format_rate,
        "avg_ttft": avg_ttft,
        "avg_total_lat": avg_total_lat,
    }


if __name__ == "__main__":
    run_unmodified_benchmark()
