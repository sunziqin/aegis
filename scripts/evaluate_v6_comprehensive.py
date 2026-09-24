# -*- coding: utf-8 -*-
"""Evaluate the V6 checkpoint on the verified state-disjoint test split.

The script deliberately keeps the data, checkpoint, and calibration artifact
bound together. A report is not written unless the selected train,
calibration, and test files have zero normalized-state overlap and the
calibration artifact names and hashes the selected checkpoint and calibration
file.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.conformal import ConformalDecisionCalibrator
from src.modeling_s1 import S1DecisionModel
from src.provenance import (
    sha256_file,
    sha256_model_directory,
    sha256_tokenizer,
    validate_provenance_pair,
)
from src.tokenizer_utils import encode_decision_batch, load_tokenizer_checked

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def path_label(path: Path, root_dir: Path = REPO_ROOT) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _require_mapping(value, label: str):
    """Turn malformed serialized artifacts into a clear fail-closed error."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON/object mapping")
    return value


def validate_declared_split_role(
    declared_split: str,
    selected_file: Path,
    root_dir: Path,
    label: str,
) -> None:
    """Check relative split labels while allowing absolute paths to move hosts."""
    declared_path = Path(declared_split)
    if declared_path.is_absolute():
        return
    try:
        selected_label = selected_file.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return
    if declared_path.as_posix() != selected_label:
        raise RuntimeError(
            f"{label} split mismatch: artifact={declared_path.as_posix()}, selected={selected_label}"
        )


def normalize_state(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def load_samples(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return payload


def dataset_manifest(path: Path, samples=None, root_dir: Path = REPO_ROOT):
    if samples is None:
        samples = load_samples(path)
    raw_states = {str(item["state"]) for item in samples}
    normalized_states = {normalize_state(item["state"]) for item in samples}
    return {
        "path": path_label(path, root_dir),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "sample_count": len(samples),
        "unique_raw_states": len(raw_states),
        "unique_normalized_states": len(normalized_states),
    }


def validate_disjoint_splits(splits):
    """Verify normalized state disjointness and return auditable provenance."""
    required = ("train", "calibration", "test")
    missing = [name for name in required if name not in splits]
    if missing:
        raise ValueError(f"Missing required split(s): {', '.join(missing)}")

    state_sets = {
        name: {normalize_state(item["state"]) for item in splits[name]}
        for name in required
    }
    intersections = {
        "train_calibration": len(state_sets["train"] & state_sets["calibration"]),
        "train_test": len(state_sets["train"] & state_sets["test"]),
        "calibration_test": len(state_sets["calibration"] & state_sets["test"]),
    }
    failures = {name: count for name, count in intersections.items() if count}
    if failures:
        detail = ", ".join(f"{name}={count}" for name, count in failures.items())
        raise ValueError(f"Normalized state overlap detected: {detail}")

    return {
        "method": "grouped_state_split",
        "normalization": "strip, lowercase, collapse whitespace",
        "verified_zero_normalized_state_overlap": True,
        "normalized_state_counts": {name: len(values) for name, values in state_sets.items()},
        "normalized_state_intersections": intersections,
    }


def validate_checkpoint_training(checkpoint, train_file: Path, root_dir: Path = REPO_ROOT):
    _require_mapping(checkpoint, "Checkpoint")
    config = checkpoint.get("config") or {}
    _require_mapping(config, "Checkpoint config")
    declared_split = config.get("train_file")
    declared_hash = str(config.get("train_data_sha256") or "").lower()
    if not declared_split or not declared_hash:
        raise RuntimeError(
            "Checkpoint has no train_file/train_data_sha256 provenance. "
            "Retrain on data/disjoint_v6/train_v6_disjoint.json before calibration or evaluation."
        )
    validate_declared_split_role(declared_split, train_file, root_dir, "Checkpoint training")
    actual_hash = sha256_file(train_file)
    if declared_hash != actual_hash:
        raise RuntimeError(
            f"Checkpoint training data hash mismatch: checkpoint={declared_hash}, selected={actual_hash}"
        )
    return actual_hash


def resolve_base_model(requested, checkpoint, root_dir: Path = REPO_ROOT) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested))
    env_value = os.environ.get("BASE_MODEL_PATH")
    if env_value:
        candidates.append(Path(env_value))
    checkpoint_base = checkpoint.get("config", {}).get("base_model")
    if checkpoint_base:
        candidates.append(Path(checkpoint_base))
    candidates.append(root_dir / "models" / "base" / "Qwen2.5-0.5B-Instruct")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    listed = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "Base model was not found. Pass --base_model or set BASE_MODEL_PATH. "
        f"Checked: {listed}"
    )


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lora_state_checked(backbone, lora_state):
    if not isinstance(lora_state, dict) or not lora_state:
        raise RuntimeError("Checkpoint backbone_lora is empty or malformed")
    expected_state = backbone.state_dict()
    unexpected = sorted(key for key in lora_state if key not in expected_state)
    shape_errors = []
    for key, value in lora_state.items():
        if not torch.is_tensor(value):
            shape_errors.append(f"{key}: value is not a tensor")
            continue
        expected = expected_state.get(key)
        if expected is not None and tuple(value.shape) != tuple(expected.shape):
            shape_errors.append(f"{key}: checkpoint={tuple(value.shape)} model={tuple(expected.shape)}")
    expected_lora_keys = {key for key in expected_state if "lora_" in key}
    actual_lora_keys = {key for key in lora_state if "lora_" in key}
    missing_lora = sorted(expected_lora_keys - actual_lora_keys)
    if unexpected or shape_errors or missing_lora:
        details = []
        if unexpected:
            details.append(f"unexpected keys={unexpected[:3]}")
        if shape_errors:
            details.append(f"shape errors={shape_errors[:3]}")
        if missing_lora:
            details.append(f"missing LoRA keys={missing_lora[:3]}")
        raise RuntimeError("Malformed LoRA checkpoint: " + "; ".join(details))
    incompatible = backbone.load_state_dict(lora_state, strict=False)
    remaining_lora = sorted(key for key in incompatible.missing_keys if "lora_" in key)
    if incompatible.unexpected_keys or remaining_lora:
        raise RuntimeError(
            "LoRA checkpoint did not load cleanly: "
            f"unexpected={incompatible.unexpected_keys[:3]}, missing={remaining_lora[:3]}"
        )


def load_s1_model(model_dir: Path, base_model: Optional[str], device: str, checkpoint=None):
    weight_file = model_dir / "s1_decision_weights.pt"
    if not weight_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_file}")

    checkpoint_sha256 = sha256_file(weight_file)
    if checkpoint is None:
        checkpoint = torch.load(weight_file, map_location=device)
    _require_mapping(checkpoint, "Checkpoint")
    base_model_path = resolve_base_model(base_model, checkpoint)
    config_provenance = checkpoint.get("config") or {}
    _require_mapping(config_provenance, "Checkpoint config")
    declared_base_model = config_provenance.get("base_model")
    if not declared_base_model:
        raise RuntimeError(
            "Checkpoint has no base_model provenance; retrain with the selected base model."
        )
    actual_base_model_sha256 = sha256_model_directory(base_model_path)
    declared_base_model_sha256 = str(config_provenance.get("base_model_sha256") or "")
    if declared_base_model_sha256.lower() != actual_base_model_sha256.lower():
        raise RuntimeError(
            "Checkpoint base_model_sha256 does not match the selected base model files; retrain the checkpoint."
        )

    tokenizer_source = model_dir if (model_dir / "tokenizer_config.json").exists() else base_model_path
    tokenizer = load_tokenizer_checked(tokenizer_source)
    actual_tokenizer_sha256 = sha256_tokenizer(tokenizer)
    declared_tokenizer_sha256 = str(checkpoint.get("config", {}).get("tokenizer_sha256") or "")
    if not _SHA256_RE.fullmatch(declared_tokenizer_sha256) or declared_tokenizer_sha256.lower() != actual_tokenizer_sha256:
        raise RuntimeError(
            "Checkpoint tokenizer_sha256 does not match the tokenizer loaded for evaluation; retrain the checkpoint."
        )
    config = AutoConfig.from_pretrained(base_model_path)
    hidden_dim = getattr(config, "hidden_size", 896)

    backbone = AutoModel.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    lora_state = checkpoint["backbone_lora"]
    sample_lora_a = next((value for key, value in lora_state.items() if "lora_A" in key), None)
    detected_r = sample_lora_a.shape[0] if sample_lora_a is not None else 32
    detected_targets = {
        module
        for key in lora_state
        for module in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if module in key
    }
    required_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    missing_targets = sorted(required_targets - detected_targets)
    if missing_targets:
        raise RuntimeError(f"Malformed V6 LoRA checkpoint: missing target modules={missing_targets}")
    target_modules = sorted(detected_targets) or ["q_proj", "k_proj", "v_proj", "o_proj"]
    backbone = get_peft_model(
        backbone,
        LoraConfig(
            r=detected_r,
            lora_alpha=detected_r * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        ),
    )

    model = S1DecisionModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_inter_layers=2,
    ).to(device)
    load_lora_state_checked(model.backbone, lora_state)
    model.decision_head.load_state_dict(checkpoint["decision_head"])
    if device == "cuda":
        model.decision_head.to(dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer, checkpoint, checkpoint_sha256, base_model_path


def load_calibration_artifact(
    calibration_file: Path,
    weight_file: Path,
    calibration_data_file: Path,
    base_model_path: Optional[Path] = None,
    base_model_sha256: Optional[str] = None,
    train_data_file: Optional[Path] = None,
    test_data_file: Optional[Path] = None,
    root_dir: Path = REPO_ROOT,
):
    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {calibration_file}")
    with calibration_file.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    _require_mapping(artifact, "Calibration artifact")

    actual_checkpoint_hash = sha256_file(weight_file)
    declared_checkpoint_hash = str(artifact.get("checkpoint_sha256") or "").lower()
    if not declared_checkpoint_hash:
        raise RuntimeError(f"Calibration artifact has no checkpoint_sha256: {calibration_file}")
    if declared_checkpoint_hash != actual_checkpoint_hash.lower():
        raise RuntimeError(
            "Calibration checkpoint hash mismatch: "
            f"artifact={declared_checkpoint_hash}, loaded={actual_checkpoint_hash}"
        )

    metadata = artifact.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Calibration artifact metadata must be a JSON object: {calibration_file}")
    declared_tokenizer_sha256 = str(metadata.get("tokenizer_sha256") or "")
    if base_model_path is not None and not _SHA256_RE.fullmatch(declared_tokenizer_sha256):
        raise RuntimeError(
            "Calibration artifact has no valid tokenizer_sha256 metadata; rerun calibration after retraining."
        )
    declared_base_model = metadata.get("base_model")
    if base_model_path is not None and not declared_base_model:
        raise RuntimeError(
            "Calibration artifact has no base_model metadata; rerun calibration after retraining."
        )
    declared_base_model_sha256 = str(metadata.get("base_model_sha256") or "")
    if base_model_path is not None and not _SHA256_RE.fullmatch(declared_base_model_sha256):
        raise RuntimeError(
            "Calibration artifact has no valid base_model_sha256 metadata; rerun calibration after retraining."
        )
    if base_model_sha256 is not None and declared_base_model_sha256.lower() != base_model_sha256.lower():
        raise RuntimeError(
            "Calibration base_model_sha256 does not match the selected base model files; rerun calibration."
        )
    declared_split = metadata.get("calibrated_split")
    if not declared_split:
        raise RuntimeError(f"Calibration artifact has no calibrated_split metadata: {calibration_file}")
    validate_declared_split_role(declared_split, calibration_data_file, root_dir, "Calibration")
    actual_data_hash = sha256_file(calibration_data_file)
    declared_data_hash = str(metadata.get("calibration_data_sha256") or "").lower()
    if not declared_data_hash:
        raise RuntimeError(
            "Calibration artifact has no calibration_data_sha256 metadata; "
            "rerun calibrate_and_test_v6.py to bind it to the selected split."
        )
    if declared_data_hash != actual_data_hash.lower():
        raise RuntimeError(
            "Calibration data hash mismatch: "
            f"artifact={declared_data_hash}, selected={actual_data_hash}"
        )

    if train_data_file is not None:
        declared_train_split = metadata.get("train_split")
        declared_train_hash = str(metadata.get("train_data_sha256") or "").lower()
        if not declared_train_split or not declared_train_hash:
            raise RuntimeError(
                "Calibration artifact has no train_split/train_data_sha256 metadata; "
                "rerun calibrate_and_test_v6.py after retraining on the disjoint train split."
            )
        validate_declared_split_role(declared_train_split, train_data_file, root_dir, "Calibration training")
        actual_train_hash = sha256_file(train_data_file)
        if declared_train_hash != actual_train_hash.lower():
            raise RuntimeError(
                "Calibration training data hash mismatch: "
                f"artifact={declared_train_hash}, selected={actual_train_hash}"
            )

    if test_data_file is not None:
        declared_test_split = metadata.get("test_split")
        declared_test_hash = str(metadata.get("test_data_sha256") or "").lower()
        if not declared_test_split or not declared_test_hash:
            raise RuntimeError(
                "Calibration artifact has no test_split/test_data_sha256 metadata; "
                "rerun calibrate_and_test_v6.py to bind the complete disjoint evaluation."
            )
        validate_declared_split_role(declared_test_split, test_data_file, root_dir, "Calibration test")
        actual_test_hash = sha256_file(test_data_file)
        if declared_test_hash != actual_test_hash.lower():
            raise RuntimeError(
                "Calibration test data hash mismatch: "
                f"artifact={declared_test_hash}, selected={actual_test_hash}"
            )

    selected_samples = load_samples(calibration_data_file)
    declared_count = int(artifact.get("num_calib_samples", -1))
    if declared_count != len(selected_samples):
        raise RuntimeError(
            f"Calibration sample count mismatch: artifact={declared_count}, selected={len(selected_samples)}"
        )
    calibrator = ConformalDecisionCalibrator()
    calibrator.load(calibration_file)
    return calibrator, artifact, actual_data_hash


class EvaluationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch_samples, tokenizer, device, max_length=2048):
    encoded = encode_decision_batch(
        tokenizer=tokenizer,
        states=[sample["state"] for sample in batch_samples],
        questions=[sample["question"] for sample in batch_samples],
        options_per_sample=[sample["candidates"] for sample in batch_samples],
        max_length=max_length,
        device=device,
    )
    encoded["targets"] = torch.tensor(
        [sample["target_idx"] for sample in batch_samples],
        dtype=torch.long,
        device=device,
    )
    encoded["domains"] = [sample.get("domain", "unknown") for sample in batch_samples]
    return encoded


def evaluate_split(model, loader, cutoff, desc=""):
    total = correct = covered = raw_act = raw_act_err = tri_act = tri_act_err = 0
    set_sizes = []
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0, "covered": 0, "tri_act": 0, "tri_act_err": 0})
    started = time.time()

    with torch.no_grad():
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
            for index, target in enumerate(targets.tolist()):
                probabilities = probs[index]
                candidate_mask = batch["marker_mask"][index].cpu().bool()
                best_index = int(torch.argmax(probabilities.masked_fill(~candidate_mask, -1.0)).item())
                confidence = float(probabilities[best_index].item())
                risk = float(risks[index].item())
                correct_prediction = best_index == target
                correct += int(correct_prediction)
                included = torch.nonzero(candidate_mask & (probabilities >= cutoff)).squeeze(-1).tolist()
                if isinstance(included, int):
                    included = [included]
                set_size = len(included)
                set_sizes.append(set_size)
                covered += int(target in included)
                is_raw_act = set_size == 1
                raw_act += int(is_raw_act)
                raw_act_err += int(is_raw_act and not correct_prediction)
                is_tri_act = is_raw_act and confidence >= 0.60 and risk <= 0.70
                tri_act += int(is_tri_act)
                tri_act_err += int(is_tri_act and not correct_prediction)

                domain = batch["domains"][index]
                stats = domain_stats[domain]
                stats["total"] += 1
                stats["correct"] += int(correct_prediction)
                stats["covered"] += int(target in included)
                stats["tri_act"] += int(is_tri_act)
                stats["tri_act_err"] += int(is_tri_act and not correct_prediction)
                total += 1

    if not total:
        raise ValueError(f"Cannot evaluate an empty split: {desc}")
    elapsed = time.time() - started
    results = {
        "description": desc,
        "sample_count": total,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_samples_per_sec": round(total / max(0.01, elapsed), 1),
        "top1_accuracy": round(correct / total, 4),
        "conformal_marginal_coverage": round(covered / total, 4),
        "raw_conformal_singleton": {
            "act_rate": round(raw_act / total, 4),
            "selective_risk": round(raw_act_err / raw_act if raw_act else 0.0, 4),
            "act_count": raw_act,
            "error_count": raw_act_err,
        },
        "production_tri_gate": {
            "act_rate": round(tri_act / total, 4),
            "selective_risk": round(tri_act_err / tri_act if tri_act else 0.0, 4),
            "act_count": tri_act,
            "error_count": tri_act_err,
        },
        "average_set_size": round(float(np.mean(set_sizes)), 3),
        "domain_breakdown": {},
    }
    for domain, stats in domain_stats.items():
        count = stats["total"]
        results["domain_breakdown"][domain] = {
            "total": count,
            "top1_accuracy": round(stats["correct"] / count, 4),
            "conformal_coverage": round(stats["covered"] / count, 4),
            "tri_gate_act_rate": round(stats["tri_act"] / count, 4),
            "tri_gate_selective_risk": round(stats["tri_act_err"] / stats["tri_act"] if stats["tri_act"] else 0.0, 4),
        }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Aegis-S1 V6 on disjoint data")
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
    calibration_data_file = args.calib_file.resolve()
    test_file = args.test_file.resolve()
    calibration_file = (args.calibration_file or model_dir / "conformal_calibration.json").resolve()
    report_file = (args.report_file or model_dir / "strictly_disjoint_evaluation_report.json").resolve()
    device = resolve_device(args.device)

    splits = {
        "train": load_samples(train_file),
        "calibration": load_samples(calibration_data_file),
        "test": load_samples(test_file),
    }
    split_provenance = validate_disjoint_splits(splits)
    data_manifests = {
        name: dataset_manifest(path, splits[name])
        for name, path in {
            "train": train_file,
            "calibration": calibration_data_file,
            "test": test_file,
        }.items()
    }

    weight_file = model_dir / "s1_decision_weights.pt"
    if not weight_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_file}")
    checkpoint = torch.load(weight_file, map_location="cpu")
    validate_checkpoint_training(checkpoint, train_file)
    base_model_path = resolve_base_model(args.base_model, checkpoint)
    base_model_sha256 = sha256_model_directory(base_model_path)
    calibrator, calibration_artifact, calibration_data_sha256 = load_calibration_artifact(
        calibration_file,
        weight_file,
        calibration_data_file,
        base_model_path=base_model_path,
        base_model_sha256=base_model_sha256,
        train_data_file=train_file,
        test_data_file=test_file,
    )
    validate_provenance_pair(checkpoint, calibration_artifact.get("metadata", {}))
    logger.info("Loading verified V6 checkpoint on %s", device)
    model, tokenizer, checkpoint, checkpoint_sha256, base_model_path = load_s1_model(
        model_dir, str(base_model_path), device, checkpoint=checkpoint
    )
    if calibration_artifact["metadata"]["tokenizer_sha256"].lower() != sha256_tokenizer(tokenizer):
        raise RuntimeError("Calibration tokenizer_sha256 does not match the tokenizer loaded for evaluation")
    q_hat = float(calibrator.quantile_threshold)
    cutoff = 1.0 - q_hat

    loader = DataLoader(
        EvaluationDataset(splits["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, device, max_length=args.max_length),
    )
    result = evaluate_split(model, loader, cutoff, desc="State-disjoint test split")
    report = {
        "schema_version": 2,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": checkpoint.get("config", {}).get("model_version", "Aegis-S1-V6"),
        "base_model": str(base_model_path),
        "base_model_sha256": base_model_sha256,
        "checkpoint": {
            "file": path_label(weight_file),
            "sha256": checkpoint_sha256,
        },
        "calibration": {
            "file": path_label(calibration_file),
            "sha256": sha256_file(calibration_file),
            "checkpoint_sha256": str(calibration_artifact["checkpoint_sha256"]).lower(),
            "calibration_data_sha256": calibration_data_sha256,
            "train_data_sha256": calibration_artifact.get("metadata", {}).get("train_data_sha256"),
            "train_split": calibration_artifact.get("metadata", {}).get("train_split"),
            "alpha": calibration_artifact.get("alpha"),
            "q_hat": q_hat,
            "num_calib_samples": calibration_artifact.get("num_calib_samples"),
            "metadata": calibration_artifact.get("metadata", {}),
        },
        "data": data_manifests,
        "split_provenance": split_provenance,
        "conformal_q_hat": q_hat,
        "conformal_cutoff_prob": cutoff,
        "splits": {"strictly_disjoint_test": result},
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info("[PASS] Evaluation report saved to %s", report_file)
    logger.info(
        "test=%d top1=%.2f%% coverage=%.2f%% tri_act=%.2f%% tri_risk=%.2f%%",
        result["sample_count"],
        result["top1_accuracy"] * 100,
        result["conformal_marginal_coverage"] * 100,
        result["production_tri_gate"]["act_rate"] * 100,
        result["production_tri_gate"]["selective_risk"] * 100,
    )
    return report


if __name__ == "__main__":
    main()
