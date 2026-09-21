# -*- coding: utf-8 -*-
"""
Run True Split-Conformal Calibration on calib_v3.json and evaluate on test_v3.json.
Uses the trained s1_model_v3 weights.
"""

import sys
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_s1 import S1DecisionDataset, collate_fn
from open_s1 import AegisRouter
from src.conformal import ConformalDecisionCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    model_dir = Path("E:/s1-decision-model/output/s1_model_v3")
    calib_path = Path("E:/s1-decision-model/data/calib_v3.json")
    test_path = Path("E:/s1-decision-model/data/test_v3.json")

    logger.info(f"[*] Loading model from {model_dir}...")
    router = AegisRouter.load(model_dir)
    model = router.model
    tokenizer = router.tokenizer
    device = router.device

    # 1. Evaluate on Calib Split
    logger.info(f"[*] Running inference on Calibration Split ({calib_path})...")
    calib_ds = S1DecisionDataset(calib_path)
    calib_loader = DataLoader(calib_ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, device))

    all_calib_probs = []
    all_calib_targets = []
    model.eval()

    with torch.no_grad():
        for b in calib_loader:
            out = model(b["input_ids"], b["attention_mask"], b["marker_indices"], b["marker_mask"])
            all_calib_probs.append(out["probs"].cpu())
            all_calib_targets.append(b["targets"].cpu())

    calib_probs = torch.cat(all_calib_probs, dim=0)
    calib_targets = torch.cat(all_calib_targets, dim=0)

    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(calib_probs, calib_targets)
    calib_file = model_dir / "conformal_calibration.json"
    calibrator.save(calib_file)
    logger.info(f"[PASS] Split-Conformal calibration complete! Fitted q_hat: {q_hat:.4f}, saved to {calib_file}")

    # 2. Evaluate on Strictly Held-Out Disjoint Test Split (ZERO template overlap)
    logger.info(f"\n[*] Evaluating on Strictly Disjoint Test Split ({test_path})...")
    test_ds = S1DecisionDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, device))

    all_test_probs = []
    all_test_targets = []

    with torch.no_grad():
        for b in test_loader:
            out = model(b["input_ids"], b["attention_mask"], b["marker_indices"], b["marker_mask"])
            all_test_probs.append(out["probs"].cpu())
            all_test_targets.append(b["targets"].cpu())

    test_probs = torch.cat(all_test_probs, dim=0)
    test_targets = torch.cat(all_test_targets, dim=0)

    test_metrics = calibrator.evaluate_coverage(test_probs, test_targets, alpha=0.05)
    test_preds = torch.argmax(test_probs, dim=-1)
    test_acc = (test_preds == test_targets).float().mean().item()

    print("\n" + "=" * 70)
    print("    UNBIASED HELD-OUT TEST EVALUATION (ZERO TEMPLATE OVERLAP)")
    print("=" * 70)
    print(f"  Test Samples:                   {len(test_ds)}")
    print(f"  Top-1 Accuracy:                 {test_acc:.2%}")
    print(f"  Empirical Conformal Coverage:   {test_metrics['empirical_coverage']:.2%} (Theoretical target: >= 95.0%)")
    print(f"  Act Coverage Rate (Auto-pass):  {test_metrics['act_coverage_rate']:.2%}")
    print(f"  Selective Risk on Act:          {test_metrics['selective_risk_on_act']:.2%}")
    print(f"  Abstention / Rejection Rate:    {test_metrics['abstention_rate']:.2%}")
    print(f"  Average Set Size:               {test_metrics['average_set_size']:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
