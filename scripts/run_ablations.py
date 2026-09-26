# -*- coding: utf-8 -*-
"""
Ablation Study Suite for Millennium-Jev.
Conducts rigorous ablation across 4 foundational dimensions on strictly state-disjoint data:
1. Gating & Trustworthiness: Full Tri-Gate vs. Partial Gates vs. Naive Softmax.
2. Representation Pooling: Option Span Mean-Pooling vs. Single Token Marker Gathering.
3. Attention Architecture: Native C++ SDPA Bidirectional Attention vs. Causal Attention & Permutation Invariance.
4. Systems Profiling: Zero-Memory SDPA vs. Explicit 4D Intermediate Mask.

Generates structured JSON report and publication-ready LaTeX tables for paper/paper_draft.tex.
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_v6_comprehensive import (
    load_s1_model,
    load_samples,
    collate_fn,
    EvaluationDataset,
    load_calibration_artifact,
    dataset_manifest,
    validate_disjoint_splits,
    sha256_file,
    sha256_model_directory,
    validate_checkpoint_training,
    validate_provenance_pair,
)
from src.tokenizer_utils import encode_decision_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation_study")


ABLATION_SEED = 1729


def _cuda_sync(device: str) -> None:
    """Synchronize only when CUDA is active; keep the benchmark runnable on CPU."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_gating_ablation(model, loader, cutoff, q_hat, device):
    """Ablation 1: Gating Policy & Conformal Safety vs Heuristics."""
    logger.info("=== Running Ablation 1: Gating Protocol & Conformal vs. Heuristics ===")
    
    samples_data = []
    
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                marker_indices=batch["marker_indices"],
                marker_mask=batch["marker_mask"],
                option_spans=batch.get("option_spans"),
                bidirectional=True,
            )
            probs = output["probs"].cpu()
            targets = batch["targets"].cpu()
            risks = output["escalate_risk"].cpu()
            
            for i, target in enumerate(targets.tolist()):
                p = probs[i]
                c_mask = batch["marker_mask"][i].cpu().bool()
                best_idx = int(torch.argmax(p.masked_fill(~c_mask, -1.0)).item())
                conf = float(p[best_idx].item())
                risk = float(risks[i].item())
                is_correct = (best_idx == target)
                
                # Conformal set
                included = torch.nonzero(c_mask & (p >= cutoff)).squeeze(-1).tolist()
                if isinstance(included, int):
                    included = [included]
                set_size = len(included)
                is_covered = target in included
                
                samples_data.append({
                    "target": target,
                    "best_idx": best_idx,
                    "is_correct": is_correct,
                    "conf": conf,
                    "risk": risk,
                    "set_size": set_size,
                    "is_covered": is_covered,
                    "domain": batch["domains"][i],
                })
                
    total = len(samples_data)
    
    # Target act rate for naive matched threshold
    full_tri_acts = sum(1 for s in samples_data if s["set_size"] == 1 and s["conf"] >= 0.60 and s["risk"] <= 0.70)
    target_act_count = full_tri_acts
    
    # Sort confidences to find matched threshold
    ranked_samples = sorted(
        enumerate(samples_data), key=lambda item: (-item[1]["conf"], item[0])
    )
    for rank, (_, sample) in enumerate(ranked_samples):
        sample["confidence_rank"] = rank
    matched_tau = (
        ranked_samples[target_act_count - 1][1]["conf"]
        if target_act_count > 0
        else 1.0
    )
    matched_threshold_count = sum(
        1 for s in samples_data if s["conf"] >= matched_tau
    )
    
    policies = {
        "Full Tri-Gate (Ours)": lambda s: s["set_size"] == 1 and s["conf"] >= 0.60 and s["risk"] <= 0.70,
        "Tri-Gate w/o Anomaly Gate": lambda s: s["set_size"] == 1 and s["conf"] >= 0.60,
        "Conformal Singleton Only": lambda s: s["set_size"] == 1,
        # Rank matching makes ties explicit instead of silently changing the
        # denominator when several examples share the threshold confidence.
        f"Naive Softmax (Matched top-N; $\\tau={matched_tau:.3f}$)": lambda s: s["confidence_rank"] < target_act_count,
        "Naive Softmax ($\\tau=0.85$)": lambda s: s["conf"] >= 0.85,
        "Naive Softmax ($\\tau=0.95$)": lambda s: s["conf"] >= 0.95,
        "Unrestricted Argmax ($\\tau=0.0$)": lambda s: True,
    }
    
    results = {}
    for name, predicate in policies.items():
        act_count = 0
        error_count = 0
        is_conformal_policy = name in {
            "Full Tri-Gate (Ours)",
            "Tri-Gate w/o Anomaly Gate",
            "Conformal Singleton Only",
        }
        covered_count = sum(1 for s in samples_data if s["is_covered"])
        
        for s in samples_data:
            if predicate(s):
                act_count += 1
                if not s["is_correct"]:
                    error_count += 1
                    
        act_rate = act_count / total
        sel_risk = (error_count / act_count) if act_count > 0 else 0.0
        act_precision = 1.0 - sel_risk
        
        results[name] = {
            "act_count": act_count,
            "act_rate": round(act_rate * 100, 2),
            "act_precision": round(act_precision * 100, 2),
            "selective_risk": round(sel_risk * 100, 2),
            "conformal_coverage": (
                round((covered_count / total) * 100, 2)
                if is_conformal_policy
                else None
            ),
            "coverage_type": "marginal_conformal" if is_conformal_policy else None,
        }
        logger.info(
            f"  {name:38s} | Act: {act_rate*100:5.2f}% | Precision: {act_precision*100:5.2f}% | Risk: {sel_risk*100:5.2f}%"
        )
        
    results["_meta"] = {
        "samples": total,
        "matched_target_act_count": target_act_count,
        "matched_threshold": round(matched_tau, 8),
        "matched_threshold_inclusive_count": matched_threshold_count,
        "matched_threshold_tie_count": max(0, matched_threshold_count - target_act_count),
    }
    return results


def run_pooling_ablation(model, loader, device, max_samples: Optional[int] = None):
    """Ablation 2: Option Span Mean-Pooling vs Single Token Marker Gathering."""
    logger.info("=== Running Ablation 2: Span Mean-Pooling vs. Single Token Marker ===")
    
    span_correct = 0
    marker_correct = 0
    total = 0
    
    multi_token_span_correct = 0
    multi_token_marker_correct = 0
    multi_token_total = 0
    
    domain_span = defaultdict(lambda: {"correct": 0, "total": 0})
    domain_marker = defaultdict(lambda: {"correct": 0, "total": 0})
    
    with torch.inference_mode():
        for batch in loader:
            if max_samples is not None and total >= max_samples:
                break

            # Slice the final batch so a bounded run reports exactly the
            # requested number of samples rather than silently overshooting.
            if max_samples is not None:
                remaining = max_samples - total
                if remaining < len(batch["targets"]):
                    batch = {
                        key: (value[:remaining] if torch.is_tensor(value) else value[:remaining])
                        for key, value in batch.items()
                    }

            targets = batch["targets"].to(device)
            c_mask = batch["marker_mask"].to(device)
            
            # 1. Full Option Span Mean-Pooling (Ours)
            out_span = model(
                batch["input_ids"],
                batch["attention_mask"],
                marker_indices=batch["marker_indices"],
                marker_mask=c_mask,
                option_spans=batch["option_spans"],
                bidirectional=True,
            )
            p_span = out_span["probs"]
            pred_span = torch.argmax(p_span.masked_fill(~c_mask, -1.0), dim=-1)
            
            # 2. Single Token Marker Gathering.  The tokenizer records the
            # actual marker position; the span end is a boundary before the
            # next marker (or sequence end), not a marker token.
            spans = batch["option_spans"]  # [B, K, 2]
            marker_indices = batch["marker_indices"].to(device)
            
            out_marker = model(
                batch["input_ids"],
                batch["attention_mask"],
                marker_indices=marker_indices,
                marker_mask=c_mask,
                option_spans=None,
                bidirectional=True,
            )
            p_marker = out_marker["probs"]
            pred_marker = torch.argmax(p_marker.masked_fill(~c_mask, -1.0), dim=-1)
            
            for j in range(len(targets)):
                t = int(targets[j].item())
                dom = batch["domains"][j]
                
                s_ok = int(pred_span[j].item() == t)
                m_ok = int(pred_marker[j].item() == t)
                
                span_correct += s_ok
                marker_correct += m_ok
                total += 1
                
                domain_span[dom]["correct"] += s_ok
                domain_span[dom]["total"] += 1
                domain_marker[dom]["correct"] += m_ok
                domain_marker[dom]["total"] += 1
                
                # Check candidate span length
                opt_lens = (spans[j, :, 1] - spans[j, :, 0]).cpu().tolist()
                if max(opt_lens) > 3: # Multi-token options
                    multi_token_span_correct += s_ok
                    multi_token_marker_correct += m_ok
                    multi_token_total += 1
                    
    if total == 0:
        raise ValueError("Pooling ablation evaluated zero samples")
    results = {
        "overall": {
            "samples": total,
            "span_pooling_acc": round((span_correct / total) * 100, 2),
            "single_marker_acc": round((marker_correct / total) * 100, 2),
            "delta": round(((span_correct - marker_correct) / total) * 100, 2),
        },
        "long_option_samples": {
            "samples": multi_token_total,
            "definition": "sample has at least one candidate span longer than three tokens",
            "span_pooling_acc": round((multi_token_span_correct / multi_token_total) * 100, 2) if multi_token_total else None,
            "single_marker_acc": round((multi_token_marker_correct / multi_token_total) * 100, 2) if multi_token_total else None,
            "delta": round(((multi_token_span_correct - multi_token_marker_correct) / multi_token_total) * 100, 2) if multi_token_total else None,
        },
        "domains": {}
    }
    
    for dom in domain_span:
        s_acc = domain_span[dom]["correct"] / domain_span[dom]["total"]
        m_acc = domain_marker[dom]["correct"] / domain_marker[dom]["total"]
        results["domains"][dom] = {
            "span_acc": round(s_acc * 100, 2),
            "marker_acc": round(m_acc * 100, 2),
            "delta": round((s_acc - m_acc) * 100, 2),
        }
        
    logger.info(f"  Overall: Span={results['overall']['span_pooling_acc']}% vs Marker={results['overall']['single_marker_acc']}% (Delta: +{results['overall']['delta']}%)")
    logger.info(f"  Multi-Token Options: Span={results['multi_token_options']['span_pooling_acc']}% vs Marker={results['multi_token_options']['single_marker_acc']}% (Delta: +{results['multi_token_options']['delta']}%)")
    return results


def run_attention_and_permutation_ablation(
    model, loader, device, max_samples: Optional[int] = None, seed: int = ABLATION_SEED
):
    """Compare attention modes and paired candidate-slot permutations.

    The permutation is applied to the candidate marker/span slots and then
    mapped back to semantic option ids.  This is a paired test: consistency is
    measured against the same sample's original prediction, while accuracy is
    measured against the correspondingly permuted target slot.
    """
    logger.info("=== Running Ablation 3: Bidirectional vs. Causal Attention & Paired Permutation ===")

    counts = {
        "bidirectional": {"correct": 0, "permuted_correct": 0, "consistent": 0, "flips": 0, "first": 0, "permuted_first": 0},
        "causal": {"correct": 0, "permuted_correct": 0, "consistent": 0, "flips": 0, "first": 0, "permuted_first": 0},
    }
    total = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def _run(batch, marker_indices, option_spans, marker_mask, bidirectional):
        return model(
            batch["input_ids"],
            batch["attention_mask"],
            marker_indices=marker_indices,
            marker_mask=marker_mask,
            option_spans=option_spans,
            bidirectional=bidirectional,
        )

    with torch.inference_mode():
        for batch in loader:
            if max_samples is not None and total >= max_samples:
                break
            targets = batch["targets"].to(device)
            marker_indices = batch["marker_indices"].to(device)
            option_spans = batch["option_spans"].to(device)
            c_mask = batch["marker_mask"].to(device)
            batch_size, max_k = c_mask.shape

            permutations = torch.arange(max_k, device=device).unsqueeze(0).repeat(batch_size, 1)
            permuted_targets = targets.clone()
            for j in range(batch_size):
                valid_k = int(c_mask[j].sum().item())
                if valid_k > 1:
                    perm = torch.randperm(valid_k, generator=generator).to(device)
                    # Ensure the paired run is actually a permutation.
                    if torch.equal(perm, torch.arange(valid_k, device=device)):
                        perm = torch.roll(perm, shifts=1, dims=0)
                    permutations[j, :valid_k] = perm
                    permuted_targets[j] = int((perm == targets[j]).nonzero(as_tuple=False)[0].item())

            permuted_marker_indices = marker_indices.gather(1, permutations)
            permuted_spans = option_spans.gather(1, permutations.unsqueeze(-1).expand(-1, -1, 2))

            for mode, bidirectional in (("bidirectional", True), ("causal", False)):
                original = _run(batch, marker_indices, option_spans, c_mask, bidirectional)
                permuted = _run(batch, permuted_marker_indices, permuted_spans, c_mask, bidirectional)
                original_pred = torch.argmax(original["probs"].masked_fill(~c_mask, -1.0), dim=-1)
                permuted_slot_pred = torch.argmax(permuted["probs"].masked_fill(~c_mask, -1.0), dim=-1)

                for j in range(batch_size):
                    valid_k = int(c_mask[j].sum().item())
                    if valid_k == 0:
                        continue
                    pred = int(original_pred[j].item())
                    perm_slot = int(permuted_slot_pred[j].item())
                    semantic_permuted_pred = int(permutations[j, perm_slot].item())
                    target = int(targets[j].item())
                    counts[mode]["correct"] += int(pred == target)
                    counts[mode]["permuted_correct"] += int(semantic_permuted_pred == target)
                    counts[mode]["consistent"] += int(semantic_permuted_pred == pred)
                    counts[mode]["flips"] += int(semantic_permuted_pred != pred)
                    counts[mode]["first"] += int(pred == 0)
                    counts[mode]["permuted_first"] += int(perm_slot == 0)
                    total += 1 if mode == "bidirectional" else 0

    if total == 0:
        raise ValueError("Attention ablation evaluated zero samples")

    def _metrics(stats):
        return {
            "top1_acc": round(stats["correct"] / total * 100, 2),
            "permuted_top1_acc": round(stats["permuted_correct"] / total * 100, 2),
            "first_index_selection_rate": round(stats["first"] / total * 100, 2),
            "option_a_selection_rate": round(stats["first"] / total * 100, 2),
            "permuted_first_index_selection_rate": round(stats["permuted_first"] / total * 100, 2),
            "permutation_consistency": round(stats["consistent"] / total * 100, 2),
            "permutation_flip_rate": round(stats["flips"] / total * 100, 2),
        }

    results = {
        "samples": total,
        "seed": seed,
        "permutation": "paired random valid-slot permutation; predictions mapped back to original option ids",
        "bidirectional": _metrics(counts["bidirectional"]),
        "causal": _metrics(counts["causal"]),
        "delta_acc": round((counts["bidirectional"]["correct"] - counts["causal"]["correct"]) / total * 100, 2),
    }
    logger.info(
        "  Bidirectional Acc: %.2f%% | consistency %.2f%% | flip %.2f%%",
        results["bidirectional"]["top1_acc"],
        results["bidirectional"]["permutation_consistency"],
        results["bidirectional"]["permutation_flip_rate"],
    )
    logger.info(
        "  Causal Acc: %.2f%% | consistency %.2f%% | flip %.2f%%",
        results["causal"]["top1_acc"],
        results["causal"]["permutation_consistency"],
        results["causal"]["permutation_flip_rate"],
    )
    return results


def run_systems_operator_ablation(model, tokenizer, device):
    """Ablation 4: fast SDPA path vs explicit 4D-mask path.

    Peak CUDA memory and latency are measured.  The explicit mask size is
    reported separately as a formula estimate because constructing a large
    mask solely for accounting would distort the operator benchmark.
    """
    logger.info("=== Running Ablation 4: Systems Latency & Memory Profile ===")
    
    seq_lengths = [512, 1024, 2048]
    batch_size = 1
    
    benchmarks = []
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device=device)
        attn_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        
        # Candidate markers
        k = 4
        marker_indices = torch.tensor([[50, 100, 150, 200]], dtype=torch.long, device=device)
        marker_mask = torch.ones((batch_size, k), dtype=torch.bool, device=device)
        option_spans = torch.tensor([[[45, 55], [95, 105], [145, 155], [195, 205]]], dtype=torch.long, device=device)
        
        # Warm up the native fast path, then measure it explicitly through
        # fast_forward (the ordinary forward path does not imply fast SDPA).
        model.enable_fast_bidirectional(True)
        for _ in range(5):
            with torch.inference_mode():
                _ = model.fast_forward(input_ids, marker_indices=marker_indices, marker_mask=marker_mask, option_spans=option_spans)

        _cuda_sync(device)
        
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
        else:
            mem_before = None
        start = time.perf_counter()
        iters = 50
        with torch.inference_mode():
            for _ in range(iters):
                _ = model.fast_forward(input_ids, marker_indices=marker_indices, marker_mask=marker_mask, option_spans=option_spans)
        _cuda_sync(device)
        sdpa_time = ((time.perf_counter() - start) / iters) * 1000
        sdpa_peak_ram = (
            (torch.cuda.max_memory_allocated() - mem_before) / (1024 * 1024)
            if mem_before is not None else None
        )

        # Measure ordinary bidirectional forward, which materializes the
        # explicit mask on this implementation.
        model.enable_fast_bidirectional(False)
        for _ in range(5):
            with torch.inference_mode():
                _ = model(input_ids, attn_mask, marker_indices=marker_indices, marker_mask=marker_mask, option_spans=option_spans, bidirectional=True)
        _cuda_sync(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            slow_mem_before = torch.cuda.memory_allocated()
        else:
            slow_mem_before = None
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(iters):
                _ = model(input_ids, attn_mask, marker_indices=marker_indices, marker_mask=marker_mask, option_spans=option_spans, bidirectional=True)
        _cuda_sync(device)
        explicit_time = ((time.perf_counter() - start) / iters) * 1000
        explicit_peak_ram = (
            (torch.cuda.max_memory_allocated() - slow_mem_before) / (1024 * 1024)
            if slow_mem_before is not None else None
        )
        
        # Formula estimate for the float32 mask; this is not a measured peak.
        explicit_4d_mask_bytes = batch_size * 1 * seq_len * seq_len * 4 # float32
        explicit_4d_mask_kb = explicit_4d_mask_bytes / 1024
        
        benchmarks.append({
            "seq_len": seq_len,
            "sdpa_latency_ms": round(sdpa_time, 2),
            "explicit_mask_latency_ms": round(explicit_time, 2),
            "sdpa_intermediate_mask_ram_kb": 0.0,
            "sdpa_mask_memory_kind": "structurally_not_allocated_by_fast_path",
            "explicit_mask_intermediate_ram_kb": round(explicit_4d_mask_kb, 1),
            "explicit_mask_memory_kind": "estimated_formula_not_measured",
            "sdpa_peak_total_mb": round(sdpa_peak_ram, 2) if sdpa_peak_ram is not None else None,
            "explicit_peak_total_mb": round(explicit_peak_ram, 2) if explicit_peak_ram is not None else None,
            "device_memory_measurement": "cuda_peak_allocated" if mem_before is not None else "unavailable_on_cpu",
        })
        logger.info(
            "  SeqLen=%d: fast=%.2fms explicit=%.2fms | mask=0 KB structural vs %.1f KB estimated",
            seq_len, sdpa_time, explicit_time, explicit_4d_mask_kb,
        )
        
    model.enable_fast_bidirectional(False)
    return benchmarks


def generate_latex_table(ablation_data: Dict[str, Any]) -> str:
    """Generate publication-ready LaTeX tables."""
    gating = ablation_data["gating"]
    pooling = ablation_data["pooling"]
    attn = ablation_data["attention"]

    def fmt(value):
        return "--" if value is None else f"{value:.2f}\\%"

    def latex_name(value):
        return value.replace("&", "\\\\&")

    gating_rows = {name: stats for name, stats in gating.items() if not name.startswith("_")}
    gating_n = gating.get("_meta", {}).get("samples", "?")
    pooling_n = pooling.get("overall", {}).get("samples", "?")
    attention_n = attn.get("samples", "?")
    
    latex_code = []
    latex_code.append("% ==========================================================")
    latex_code.append("% TABLE 2: COMPREHENSIVE ABLATION STUDY ACROSS FOUR PILLARS")
    latex_code.append("% ==========================================================")
    latex_code.append("\\begin{table*}[t]")
    latex_code.append("\\centering")
    latex_code.append("\\small")
    latex_code.append("\\begin{subtable}[t]{\\textwidth}")
    latex_code.append("\\centering")
    latex_code.append("\\begin{tabular}{lcccc}")
    latex_code.append("\\toprule")
    latex_code.append("\\textbf{Gating Protocol} & \\textbf{Act Clearance (\\%)} & \\textbf{Act Precision (\\%)} & \\textbf{Selective Risk (\\%)} & \\textbf{Coverage (\\%)} \\\\")
    latex_code.append("\\midrule")
    
    for name, stats in gating_rows.items():
        bold_prefix = "\\textbf{" if "Full Tri-Gate" in name else ""
        bold_suffix = "}" if "Full Tri-Gate" in name else ""
        latex_code.append(
            f"{bold_prefix}{latex_name(name)}{bold_suffix} & {fmt(stats['act_rate'])} & {fmt(stats['act_precision'])} & {fmt(stats['selective_risk'])} & {fmt(stats['conformal_coverage'])} \\\\"
        )
        
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\caption{Ablation 2A: Gating Policy \\& Trustworthy Decision Making under Conformal Boundaries.}")
    latex_code.append("\\label{tab:ablation_gating}")
    latex_code.append("\\end{subtable}")
    latex_code.append("")
    latex_code.append("\\vspace{2mm}")
    latex_code.append("\\begin{subtable}[t]{\\textwidth}")
    latex_code.append("\\centering")
    latex_code.append("\\begin{tabular}{lcccc}")
    latex_code.append("\\toprule")
    latex_code.append("\\textbf{Architectural Component} & \\textbf{Configuration} & \\textbf{Top-1 Accuracy} & \\textbf{Multi-Token Acc} & \\textbf{Position $A$-Bias} \\\\")
    latex_code.append("\\midrule")
    
    # Rows for components. Long-option metrics are a sample-level subset
    # summary, not a token-level accuracy claim.
    long_options = pooling.get("long_option_samples", pooling.get("multi_token_options", {}))
    latex_code.append(f"\\textbf{{Full Millennium-Jev (V6)}} & Span Pooling + SDPA & \\textbf{{{pooling['overall']['span_pooling_acc']:.2f}\\%}} & {fmt(long_options.get('span_pooling_acc'))} & \\textbf{{{attn['bidirectional']['option_a_selection_rate']:.2f}\\%}} \\\\")
    latex_code.append(f"w/o Span Pooling & Single Token Marker & {pooling['overall']['single_marker_acc']:.2f}\\% ($-{pooling['overall']['delta']:.2f}\\%$) & {fmt(long_options.get('single_marker_acc'))} & {attn['bidirectional']['option_a_selection_rate']:.2f}\\% \\\\")
    latex_code.append(f"w/o Bidirectional Attention & Lower-Triangular Causal & {attn['causal']['top1_acc']:.2f}\\% ($-{attn['delta_acc']:.2f}\\%$) & -- & {attn['causal']['option_a_selection_rate']:.2f}\\% ($+{attn['causal']['option_a_selection_rate'] - attn['bidirectional']['option_a_selection_rate']:.2f}\\%$) \\\\")
    
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\caption{Ablation 2B: Architectural Component Breakdown on Disjoint Representation Learning.}")
    latex_code.append("\\label{tab:ablation_architecture}")
    latex_code.append("\\end{subtable}")
    latex_code.append(f"\\caption{{Comprehensive Ablation Studies. Gating $N={gating_n}$; pooling $N={pooling_n}$; paired attention $N={attention_n}$. Results use the verified state-disjoint test split.}}")
    latex_code.append("\\label{tab:main_ablation}")
    latex_code.append("\\end{table*}")
    
    return "\n".join(latex_code)


def main():
    model_dir = REPO_ROOT / "output" / "s1_model_v6"
    base_model_path = REPO_ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct"
    calib_file = REPO_ROOT / "data" / "disjoint_v6" / "calib_v6_disjoint.json"
    train_file = REPO_ROOT / "data" / "disjoint_v6" / "train_v6_disjoint.json"
    test_file = REPO_ROOT / "data" / "disjoint_v6" / "test_v6_disjoint.json"
    calibration_artifact_file = model_dir / "conformal_calibration.json"
    
    torch.manual_seed(ABLATION_SEED)
    np.random.seed(ABLATION_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting Ablation Study Suite on {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})...")
    
    # 1. Load Calibration
    base_model_sha256 = sha256_model_directory(base_model_path)
    calibrator, calib_artifact, calibration_data_sha256 = load_calibration_artifact(
        calibration_artifact_file,
        model_dir / "s1_decision_weights.pt",
        calib_file,
        base_model_path=base_model_path,
        base_model_sha256=base_model_sha256,
        train_data_file=train_file,
        test_data_file=test_file,
    )
    q_hat = float(calibrator.quantile_threshold)
    cutoff = 1.0 - q_hat
    logger.info(f"Conformal Threshold: q_hat={q_hat:.4f} (cutoff={cutoff:.4f})")
    
    # 2. Load Model & Tokenizer
    model, tokenizer, checkpoint, _, _ = load_s1_model(
        model_dir, str(base_model_path), device
    )
    checkpoint_file = model_dir / "s1_decision_weights.pt"
    checkpoint_sha256 = sha256_file(checkpoint_file)
    validate_checkpoint_training(checkpoint, train_file)
    validate_provenance_pair(checkpoint, calib_artifact.get("metadata", {}))
    model.eval()
    
    # 3. Load Test Data
    samples = load_samples(test_file)
    logger.info(f"Loaded {len(samples)} strictly state-disjoint test samples.")
    split_samples = {
        "train": load_samples(train_file),
        "calibration": load_samples(calib_file),
        "test": samples,
    }
    split_provenance = validate_disjoint_splits(split_samples)
    data_manifests = {
        name: dataset_manifest(path, split_samples[name])
        for name, path in {
            "train": train_file,
            "calibration": calib_file,
            "test": test_file,
        }.items()
    }
    
    loader = DataLoader(
        EvaluationDataset(samples),
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
    )
    
    # Run all ablations
    gating_results = run_gating_ablation(model, loader, cutoff, q_hat, device)
    pooling_results = run_pooling_ablation(model, loader, device, max_samples=None)
    attention_results = run_attention_and_permutation_ablation(model, loader, device, max_samples=None)
    systems_results = run_systems_operator_ablation(model, tokenizer, device)
    
    ablation_payload = {
        "schema_version": 2,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "seed": ABLATION_SEED,
        "provenance": {
            "model_version": checkpoint.get("config", {}).get("model_version", "Aegis-S1-V6"),
            "checkpoint": {
                "file": str(checkpoint_file.relative_to(REPO_ROOT)).replace(os.sep, "/"),
                "sha256": checkpoint_sha256,
            },
            "base_model": str(base_model_path),
            "base_model_sha256": base_model_sha256,
            "calibration": {
                "file": str(calibration_artifact_file.relative_to(REPO_ROOT)).replace(os.sep, "/"),
                "sha256": sha256_file(calibration_artifact_file),
                "data_sha256": calibration_data_sha256,
                "q_hat": q_hat,
                "metadata": calib_artifact.get("metadata", {}),
            },
            "data": data_manifests,
            "split_provenance": split_provenance,
        },
        "conformal_q_hat": q_hat,
        "conformal_cutoff": cutoff,
        "gating": gating_results,
        "pooling": pooling_results,
        "attention": attention_results,
        "systems": systems_results,
    }
    
    out_json = REPO_ROOT / "output" / "ablation_study_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(ablation_payload, f, indent=2, ensure_ascii=False)
    logger.info(f"[SUCCESS] Ablation report saved to {out_json}")
    
    latex_table = generate_latex_table(ablation_payload)
    out_tex = REPO_ROOT / "paper" / "table2_ablation.tex"
    with out_tex.open("w", encoding="utf-8") as f:
        f.write(latex_table)
    logger.info(f"[SUCCESS] LaTeX Table 2 saved to {out_tex}")
    
    print("\n" + "=" * 70)
    print("               GENERATED PUBLICATION-READY LATEX TABLE")
    print("=" * 70)
    print(latex_table)
    print("=" * 70)


if __name__ == "__main__":
    main()
