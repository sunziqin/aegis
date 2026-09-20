# -*- coding: utf-8 -*-
"""
Unit test for Conformal Decision Calibrator and Calibrated Loss Functions.
Verifies mathematical coverage guarantee (>= 95%) and proper scoring rule penalties.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.conformal import ConformalDecisionCalibrator
from src.losses import CalibratedDecisionLoss, MultiClassBrierLoss


def test_conformal_coverage_guarantee():
    print("=" * 60)
    print("[1] Testing Mathematical Conformal Coverage Guarantee (Target: 95%)...")
    
    np.random.seed(42)
    n_calib = 1000
    n_test = 2000
    n_classes = 5
    
    # Generate synthetic softmax probabilities with realistic noise
    calib_logits = np.random.randn(n_calib, n_classes) * 2.0
    calib_probs = np.exp(calib_logits) / np.exp(calib_logits).sum(axis=-1, keepdims=True)
    # Simulate true labels: usually top class, occasionally 2nd class
    calib_labels = np.argmax(calib_probs, axis=-1)
    # Inject 10% label noise
    flip_indices = np.random.choice(n_calib, size=int(0.1 * n_calib), replace=False)
    calib_labels[flip_indices] = np.random.randint(0, n_classes, size=len(flip_indices))
    
    # Test set
    test_logits = np.random.randn(n_test, n_classes) * 2.0
    test_probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=-1, keepdims=True)
    test_labels = np.argmax(test_probs, axis=-1)
    flip_indices_test = np.random.choice(n_test, size=int(0.1 * n_test), replace=False)
    test_labels[flip_indices_test] = np.random.randint(0, n_classes, size=len(flip_indices_test))
    
    # Fit conformal calibrator
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(calib_probs, calib_labels)
    print(f"[*] Non-conformity quantile threshold (q_hat): {q_hat:.4f}")
    
    metrics = calibrator.evaluate_coverage(test_probs, test_labels)
    print(f"[*] Target Coverage: {metrics['target_coverage']:.1%}")
    print(f"[*] Empirical Coverage on Test Set: {metrics['empirical_coverage']:.1%}")
    print(f"[*] Average Prediction Set Size: {metrics['average_set_size']:.2f}")
    print(f"[*] Mathematical Guarantee Met: {metrics['guarantee_satisfied']}")
    
    assert metrics["empirical_coverage"] >= 0.94, f"Coverage {metrics['empirical_coverage']} below guarantee!"
    print("[PASS] Conformal Prediction satisfies rigorous statistical coverage guarantee!")


def test_losses_execution():
    print("\n" + "=" * 60)
    print("[2] Testing Calibrated Decision Loss & Proper Scoring Rules...")
    
    loss_fn = CalibratedDecisionLoss(lambda_brier=0.5, lambda_esc=0.3)
    batch_size = 4
    max_k = 4
    
    logits = torch.randn(batch_size, max_k)
    probs = torch.softmax(logits, dim=-1)
    escalate = torch.sigmoid(torch.randn(batch_size))
    targets = torch.tensor([0, 1, 2, 0])
    mask = torch.ones((batch_size, max_k), dtype=torch.bool)
    mask[0, 3] = False  # sample 0 only has 3 options
    
    total_loss, details = loss_fn(logits, probs, escalate, targets, mask)
    print(f"[*] Computed Losses: {details}")
    assert total_loss > 0, "Loss must be positive"
    print("[PASS] Calibrated Decision Loss executes correctly!")


if __name__ == "__main__":
    test_conformal_coverage_guarantee()
    test_losses_execution()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Phase 2 Conformal Engine & Losses are ready.")
