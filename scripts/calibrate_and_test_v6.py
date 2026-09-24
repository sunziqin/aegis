# -*- coding: utf-8 -*-
"""Calibrate and evaluate a V6 checkpoint on the disjoint data contract.

The default calibration and test files are the state-grouped files under
``data/disjoint_v6``. The script refuses checkpoints without matching training
data provenance and writes calibration/report artifacts bound to file hashes.
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_v6_comprehensive import (
    dataset_manifest,
    load_samples,
    load_s1_model,
    path_label,
    resolve_base_model,
    resolve_device,
    sha256_file,
    validate_checkpoint_training,
    validate_disjoint_splits,
)
from scripts.train_s1 import S1DecisionDataset, collate_fn, sha256_tokenizer
from src.conformal import ConformalDecisionCalibrator
from src.provenance import sha256_model_directory, validate_provenance_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def collect_outputs(model, loader):
    probabilities = []
    targets = []
    risks = []
    masks = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                marker_indices=batch["marker_indices"],
                marker_mask=batch["marker_mask"],
                option_spans=batch.get("option_spans"),
            )
            probabilities.append(output["probs"].detach().cpu().float())
            targets.append(batch["targets"].detach().cpu().long())
            risks.append(output["escalate_risk"].detach().cpu().float())
            masks.append(batch["marker_mask"].detach().cpu().bool())
    if not targets:
        raise ValueError("Cannot calibrate or evaluate an empty split")
    max_options = max(chunk.shape[1] for chunk in probabilities)
    padded_probabilities = []
    padded_masks = []
    for chunk in probabilities:
        chunk_mask = masks[len(padded_probabilities)]
        if chunk.shape[1] < max_options:
            padding = torch.zeros(
                (chunk.shape[0], max_options - chunk.shape[1]),
                dtype=chunk.dtype,
            )
            chunk = torch.cat((chunk, padding), dim=1)
            chunk_mask = torch.cat(
                (
                    chunk_mask,
                    torch.zeros(
                        (chunk_mask.shape[0], max_options - chunk_mask.shape[1]),
                        dtype=torch.bool,
                    ),
                ),
                dim=1,
            )
        padded_probabilities.append(chunk)
        padded_masks.append(chunk_mask)
    return torch.cat(padded_probabilities), torch.cat(targets), torch.cat(risks), torch.cat(padded_masks)


def evaluate_outputs(probabilities, targets, risks, valid_masks, cutoff, samples):
    total = len(targets)
    correct = covered = raw_act = raw_errors = tri_act = tri_errors = 0
    set_sizes = []
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0, "covered": 0, "tri_act": 0, "tri_errors": 0})
    for index in range(total):
        probs = probabilities[index]
        candidate_mask = valid_masks[index]
        target = int(targets[index].item())
        best = int(torch.argmax(probs.masked_fill(~candidate_mask, -1.0)).item())
        correct_prediction = best == target
        correct += int(correct_prediction)
        included = torch.nonzero(candidate_mask & (probs >= cutoff)).squeeze(-1).tolist()
        if isinstance(included, int):
            included = [included]
        set_size = len(included)
        set_sizes.append(set_size)
        covered += int(target in included)
        is_raw_act = set_size == 1
        raw_act += int(is_raw_act)
        raw_errors += int(is_raw_act and not correct_prediction)
        is_tri_act = is_raw_act and float(probs[best]) >= 0.60 and float(risks[index]) <= 0.70
        tri_act += int(is_tri_act)
        tri_errors += int(is_tri_act and not correct_prediction)

        domain = samples[index].get("domain", "unknown")
        stats = domain_stats[domain]
        stats["total"] += 1
        stats["correct"] += int(correct_prediction)
        stats["covered"] += int(target in included)
        stats["tri_act"] += int(is_tri_act)
        stats["tri_errors"] += int(is_tri_act and not correct_prediction)

    result = {
        "sample_count": total,
        "top1_accuracy": round(correct / total, 4),
        "conformal_marginal_coverage": round(covered / total, 4),
        "raw_conformal_singleton": {
            "act_rate": round(raw_act / total, 4),
            "selective_risk": round(raw_errors / raw_act if raw_act else 0.0, 4),
            "act_count": raw_act,
            "error_count": raw_errors,
        },
        "production_tri_gate": {
            "act_rate": round(tri_act / total, 4),
            "selective_risk": round(tri_errors / tri_act if tri_act else 0.0, 4),
            "act_count": tri_act,
            "error_count": tri_errors,
        },
        "average_set_size": round(float(np.mean(set_sizes)), 3),
        "domain_breakdown": {},
    }
    for domain, stats in domain_stats.items():
        count = stats["total"]
        result["domain_breakdown"][domain] = {
            "total": count,
            "top1_accuracy": round(stats["correct"] / count, 4),
            "conformal_coverage": round(stats["covered"] / count, 4),
            "tri_gate_act_rate": round(stats["tri_act"] / count, 4),
            "tri_gate_selective_risk": round(stats["tri_errors"] / stats["tri_act"] if stats["tri_act"] else 0.0, 4),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate and test Aegis-S1 V6 on disjoint data")
    parser.add_argument("--model_dir", type=Path, default=REPO_ROOT / "output" / "s1_model_v6")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--train_file", type=Path, default=REPO_ROOT / "data" / "disjoint_v6" / "train_v6_disjoint.json")
    parser.add_argument("--calib_file", type=Path, default=REPO_ROOT / "data" / "disjoint_v6" / "calib_v6_disjoint.json")
    parser.add_argument("--test_file", type=Path, default=REPO_ROOT / "data" / "disjoint_v6" / "test_v6_disjoint.json")
    parser.add_argument("--calibration_file", type=Path, default=None)
    parser.add_argument("--report_file", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    train_file = args.train_file.resolve()
    calib_file = args.calib_file.resolve()
    test_file = args.test_file.resolve()
    calibration_file = (args.calibration_file or model_dir / "conformal_calibration.json").resolve()
    report_file = (args.report_file or model_dir / "test_evaluation_report.json").resolve()
    device = resolve_device(args.device)

    splits = {
        "train": load_samples(train_file),
        "calibration": load_samples(calib_file),
        "test": load_samples(test_file),
    }
    split_provenance = validate_disjoint_splits(splits)
    data_manifests = {
        name: dataset_manifest(path, splits[name])
        for name, path in {"train": train_file, "calibration": calib_file, "test": test_file}.items()
    }

    weight_file = model_dir / "s1_decision_weights.pt"
    if not weight_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_file}")
    checkpoint = torch.load(weight_file, map_location="cpu")
    train_data_sha256 = validate_checkpoint_training(checkpoint, train_file)
    base_model_path = resolve_base_model(args.base_model, checkpoint)
    base_model_sha256 = sha256_model_directory(base_model_path)
    logger.info("Loading verified V6 checkpoint on %s", device)
    model, tokenizer, checkpoint, checkpoint_sha256, base_model_path = load_s1_model(
        model_dir, str(base_model_path), device, checkpoint=checkpoint
    )
    tokenizer_sha256 = sha256_tokenizer(tokenizer)
    declared_tokenizer_sha256 = str(checkpoint.get("config", {}).get("tokenizer_sha256") or "").lower()
    if not declared_tokenizer_sha256 or declared_tokenizer_sha256 != tokenizer_sha256.lower():
        raise RuntimeError(
            "Checkpoint tokenizer_sha256 does not match the tokenizer loaded for calibration; retrain the checkpoint."
        )

    calib_dataset = S1DecisionDataset(calib_file)
    test_dataset = S1DecisionDataset(test_file)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, device, shuffle_options=False, max_length=args.max_length),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, device, shuffle_options=False, max_length=args.max_length),
    )

    started = time.time()
    calib_probs, calib_targets, _, _ = collect_outputs(model, calib_loader)
    test_probs, test_targets, test_risks, test_masks = collect_outputs(model, test_loader)
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(calib_probs, calib_targets)
    cutoff = 1.0 - q_hat
    test_result = evaluate_outputs(test_probs, test_targets, test_risks, test_masks, cutoff, splits["test"])
    elapsed = time.time() - started
    calib_accuracy = float((torch.argmax(calib_probs, dim=-1) == calib_targets).float().mean().item())

    calibration_metadata = {
        "model_version": checkpoint.get("config", {}).get("model_version", "Aegis-S1-V6"),
        "base_model": str(base_model_path),
        "base_model_sha256": base_model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "calibrated_split": path_label(calib_file),
        "calibrated_samples": len(splits["calibration"]),
        "calibration_data_sha256": data_manifests["calibration"]["sha256"],
        "train_split": path_label(train_file),
        "train_data_sha256": train_data_sha256,
        "test_split": path_label(test_file),
        "test_data_sha256": data_manifests["test"]["sha256"],
        "split_provenance": split_provenance,
        "bound_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_provenance_pair(checkpoint, calibration_metadata)
    model_dir.mkdir(parents=True, exist_ok=True)
    calibrator.save(calibration_file, checkpoint_sha256=checkpoint_sha256, metadata=calibration_metadata)

    report = {
        "schema_version": 2,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "model_version": checkpoint.get("config", {}).get("model_version", "Aegis-S1-V6"),
        "base_model": str(base_model_path),
        "base_model_sha256": base_model_sha256,
        "checkpoint": {"file": path_label(weight_file), "sha256": checkpoint_sha256},
        "calibration": {
            "file": path_label(calibration_file),
            "sha256": sha256_file(calibration_file),
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_data_sha256": data_manifests["calibration"]["sha256"],
            "alpha": calibrator.alpha,
            "q_hat": q_hat,
            "num_calib_samples": len(splits["calibration"]),
            "accuracy": round(calib_accuracy, 4),
            "metadata": calibration_metadata,
        },
        "data": data_manifests,
        "split_provenance": split_provenance,
        "test": test_result,
        "test_samples": test_result["sample_count"],
        "top1_accuracy": test_result["top1_accuracy"],
        "conformal_coverage": test_result["conformal_marginal_coverage"],
        "raw_conformal_singleton": test_result["raw_conformal_singleton"],
        "production_tri_gate": test_result["production_tri_gate"],
        "avg_prediction_set_size": test_result["average_set_size"],
        "q_hat": q_hat,
        "cutoff_prob": cutoff,
        "calib_accuracy": round(calib_accuracy, 4),
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info("[PASS] Calibration and evaluation artifacts saved under %s", model_dir)
    logger.info(
        "calib=%d test=%d q_hat=%.6f top1=%.2f%% coverage=%.2f%% tri_risk=%.2f%%",
        len(splits["calibration"]),
        test_result["sample_count"],
        q_hat,
        test_result["top1_accuracy"] * 100,
        test_result["conformal_marginal_coverage"] * 100,
        test_result["production_tri_gate"]["selective_risk"] * 100,
    )
    return report


if __name__ == "__main__":
    main()
