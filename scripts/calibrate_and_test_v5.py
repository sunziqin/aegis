# -*- coding: utf-8 -*-
"""
Split-Conformal Calibration and Held-out Test Evaluation for Aegis-S1 V5.
Loads the trained checkpoint from output/s1_model_v5/s1_decision_weights.pt,
fits conformal threshold q_hat on calib_v5.json,
saves conformal_calibration.json,
and evaluates coverage & accuracy on a held-out test_v5.json split.
"""

import json
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling_s1 import S1DecisionModel
from src.tokenizer_utils import encode_decision_batch
from src.conformal import ConformalDecisionCalibrator
from scripts.train_s1 import S1DecisionDataset, collate_fn
from scripts.evaluate_v6_comprehensive import load_lora_state_checked

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_calibration_and_test():
    base_model = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
    model_dir = Path("E:/s1-decision-model/output/s1_model_v5")
    weight_file = model_dir / "s1_decision_weights.pt"
    calib_file = Path("E:/s1-decision-model/data/calib_v5.json")
    test_file = Path("E:/s1-decision-model/data/test_v5.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer & Config
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(base_model)
    hidden_dim = getattr(config, "hidden_size", 896)

    # 2. Build Backbone + LoRA
    backbone = AutoModel.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)

    # 3. Build Model & Load Weights
    model = S1DecisionModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_inter_layers=2,
    ).to(device)

    if device == "cuda":
        model.decision_head.to(dtype=torch.bfloat16)

    logger.info(f"[*] Loading trained weights from: {weight_file}")
    ckpt = torch.load(weight_file, map_location=device)
    load_lora_state_checked(model.backbone, ckpt["backbone_lora"])
    model.decision_head.load_state_dict(ckpt["decision_head"])
    model.eval()
    logger.info(f"[PASS] Model V5 loaded successfully!")

    # 4. Prepare Calib and Test Loaders
    calib_dataset = S1DecisionDataset(calib_file)
    test_dataset = S1DecisionDataset(test_file)

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, device, shuffle_options=False, max_length=2048),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, device, shuffle_options=False, max_length=2048),
    )

    # 5. Split-Conformal Calibration on Calib Set
    logger.info(f"[*] Running Split-Conformal Calibration on Calib Split ({len(calib_dataset)} samples)...")
    calib_probs = []
    calib_targets = []
    
    with torch.no_grad():
        for b in calib_loader:
            out = model(
                b["input_ids"],
                b["attention_mask"],
                marker_indices=b["marker_indices"],
                marker_mask=b["marker_mask"],
                option_spans=b.get("option_spans"),
            )
            calib_probs.append(out["probs"].detach().cpu().float())
            calib_targets.append(b["targets"].detach().cpu().long())

    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(torch.cat(calib_probs), torch.cat(calib_targets))
    
    calib_file_path = model_dir / "conformal_calibration.json"
    calibrator.save(
        calib_file_path,
        checkpoint_sha256=sha256_file(weight_file),
        metadata={
            "model_version": "Aegis-S1-V5-legacy",
            "calibrated_split": str(calib_file),
            "calibration_data_sha256": sha256_file(calib_file),
            "test_split": str(test_file),
            "test_data_sha256": sha256_file(test_file),
            "legacy_provenance": True,
        },
    )
    logger.info(f"[PASS] Calibration fitted: q_hat={q_hat:.4f} (cut-off prob: {1.0 - q_hat:.4f}), saved to {calib_file_path}")

    # 6. Unbiased Evaluation on the held-out test split
    logger.info(f"[*] Evaluating on Disjoint Test Split ({len(test_dataset)} samples)...")
    total_samples = 0
    correct_count = 0
    covered_count = 0
    act_count = 0
    act_errors = 0
    set_sizes = []

    cutoff = 1.0 - q_hat

    with torch.no_grad():
        for b in test_loader:
            out = model(
                b["input_ids"],
                b["attention_mask"],
                marker_indices=b["marker_indices"],
                marker_mask=b["marker_mask"],
                option_spans=b.get("option_spans"),
            )
            probs = out["probs"].cpu()
            targets = b["targets"].cpu()

            for i in range(len(targets)):
                p = probs[i]
                t_idx = targets[i].item()
                best_idx = torch.argmax(p).item()

                if best_idx == t_idx:
                    correct_count += 1

                in_set = torch.nonzero(p >= cutoff).squeeze(-1).tolist()
                if isinstance(in_set, int):
                    in_set = [in_set]
                set_size = len(in_set)
                set_sizes.append(set_size)

                if t_idx in in_set:
                    covered_count += 1

                if set_size == 1:
                    act_count += 1
                    if best_idx != t_idx:
                        act_errors += 1

                total_samples += 1

    top1_acc = correct_count / total_samples
    coverage_rate = covered_count / total_samples
    act_rate = act_count / total_samples
    selective_risk = (act_errors / act_count) if act_count > 0 else 0.0
    avg_set_size = float(np.mean(set_sizes))

    logger.info("\n" + "=" * 65)
    logger.info("    AEGIS-S1 V5 UNBIASED HELD-OUT TEST EVALUATION")
    logger.info("=" * 65)
    logger.info(f"  Test Sample Count:              {total_samples}")
    logger.info(f"  Top-1 Accuracy:                 {top1_acc:.2%}")
    logger.info(f"  Empirical Conformal Coverage:   {coverage_rate:.2%} (Theoretical Target: >= 95.0%)")
    logger.info(f"  Act Coverage Rate (Auto-pass):  {act_rate:.2%}")
    logger.info(f"  Selective Risk on Act:          {selective_risk:.2%}")
    logger.info(f"  Average Prediction Set Size:    {avg_set_size:.2f}")
    logger.info("=" * 65)

    # Save tokenizer files to model_dir as well
    tokenizer.save_pretrained(model_dir)
    logger.info(f"[PASS] Tokenizer saved to: {model_dir}")


if __name__ == "__main__":
    run_calibration_and_test()
