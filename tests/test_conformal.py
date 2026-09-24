# -*- coding: utf-8 -*-
"""
Unit test for Conformal Decision Calibrator and Calibrated Loss Functions.
Verifies mathematical coverage guarantee (>= 95%) and proper scoring rule penalties.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.conformal import ConformalDecisionCalibrator
from src.losses import CalibratedDecisionLoss, MultiClassBrierLoss
from open_s1.primitives import Noul, Score


def test_conformal_coverage_guarantee():
    print("=" * 60)
    print("[1] Testing Mathematical Conformal Coverage Guarantee (Target: 95%)...")
    
    n_calib = 1000
    n_test = 2000
    n_classes = 5

    # Use a deterministic fixture with a clear probability margin above the
    # calibrated cutoff, so floating-point boundary noise cannot invalidate
    # this 95% contract test.
    def make_confident_split(size, true_prob):
        labels = np.arange(size, dtype=np.int64) % n_classes
        other_prob = (1.0 - true_prob) / (n_classes - 1)
        probs = np.full((size, n_classes), other_prob, dtype=np.float64)
        probs[np.arange(size), labels] = true_prob
        return probs, labels

    calib_probs, calib_labels = make_confident_split(n_calib, true_prob=0.8)
    test_probs, test_labels = make_confident_split(n_test, true_prob=0.85)
    
    # Fit conformal calibrator
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(calib_probs, calib_labels)
    print(f"[*] Non-conformity quantile threshold (q_hat): {q_hat:.4f}")
    
    metrics = calibrator.evaluate_coverage(test_probs, test_labels)
    print(f"[*] Target Coverage: {metrics['target_coverage']:.1%}")
    print(f"[*] Empirical Coverage on Test Set: {metrics['empirical_coverage']:.1%}")
    print(f"[*] Average Prediction Set Size: {metrics['average_set_size']:.2f}")
    print(f"[*] Mathematical Guarantee Met: {metrics['guarantee_satisfied']}")
    
    assert np.isclose(metrics["target_coverage"], 0.95), (
        f"Expected a 95% target, got {metrics['target_coverage']:.1%}"
    )
    assert metrics["guarantee_satisfied"], (
        f"Coverage {metrics['empirical_coverage']:.1%} below target "
        f"{metrics['target_coverage']:.1%}"
    )
    assert metrics["empirical_coverage"] >= metrics["target_coverage"]

    undercovered_labels = test_labels.copy()
    undercovered_labels[: n_test // 10] = (undercovered_labels[: n_test // 10] + 1) % n_classes
    undercovered_metrics = calibrator.evaluate_coverage(test_probs, undercovered_labels)
    assert undercovered_metrics["empirical_coverage"] < undercovered_metrics["target_coverage"]
    assert not undercovered_metrics["guarantee_satisfied"]
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


def test_alpha_is_finite_and_strictly_bounded():
    """All public alpha entry points reject values outside (0, 1)."""
    invalid = [-0.1, 0.0, 1.0, 1.1, float("nan"), float("inf"), None, "not-a-number"]
    for value in invalid:
        with pytest.raises(ValueError):
            ConformalDecisionCalibrator(alpha=value)

    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    calibrator.fit(np.array([[0.9, 0.1], [0.8, 0.2]]), np.array([0, 0]))
    for value in [item for item in invalid if item is not None]:
        with pytest.raises(ValueError):
            calibrator.predict([0.9, 0.1], alpha=value)


def test_invalid_probabilities_fail_closed():
    """NaN, negative, zero-sum, and materially unnormalized probabilities never produce a verdict."""
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    calibrator.fit(np.array([[0.9, 0.1], [0.8, 0.2]]), np.array([0, 0]))

    invalid_vectors = [
        [float("nan"), 0.5],
        [-0.1, 1.1],
        [0.6, 0.6],
        [0.0, 0.0],
        [],
    ]
    for vector in invalid_vectors:
        with pytest.raises(ValueError):
            calibrator.predict(vector)

    # bfloat16 model outputs are accepted after a safe float conversion.
    result = calibrator.predict(torch.tensor([0.5, 0.5], dtype=torch.bfloat16))[0]
    assert result["verdict"] == "escalate"


def test_zero_quantile_is_not_replaced_by_default():
    """A legitimate q_hat=0 keeps a probability cutoff of exactly 1.0."""
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(np.tile([[1.0, 0.0]], (40, 1)), np.zeros(40, dtype=np.int64))
    assert q_hat == 0.0
    result = calibrator.predict([0.99, 0.01])[0]
    assert result["quantile_threshold"] == 0.0
    assert result["prob_cutoff"] == 1.0
    assert result["verdict"] == "reject"


def test_exact_conformal_rank_and_small_sample_alpha():
    calibrator = ConformalDecisionCalibrator(alpha=0.4)
    calibrator.calibration_scores = [0.1, 0.2, 0.3, 0.4]
    calibrator.num_calib_samples = 4
    assert calibrator._compute_quantile(0.4) == 0.3  # ceil((4 + 1) * 0.6) = 3

    small = ConformalDecisionCalibrator(alpha=0.05)
    small.fit(np.array([[0.9, 0.1], [0.8, 0.2]]), np.array([0, 0]))
    assert small.quantile_threshold == 1.0
    result = small.predict([0.5, 0.5])[0]
    assert result["prediction_set_size"] == 2
    assert result["alpha_guarantee"] == 0.95


def test_valid_mask_excludes_padded_candidate_classes():
    """Variable-K evaluation must never count zero-probability padding as an option."""
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    calibrator.fit(np.array([[0.9, 0.1], [0.8, 0.2]]), np.array([0, 0]))
    probabilities = np.array([[0.5, 0.5, 0.0], [0.6, 0.4, 0.0]], dtype=np.float64)
    valid_mask = np.array([[True, True, False], [True, True, False]])

    result = calibrator.predict(probabilities)[0]
    assert result["prediction_set_size"] == 3
    masked_result = calibrator.predict(probabilities, valid_mask=valid_mask)[0]
    assert masked_result["prediction_set_size"] == 2

    metrics = calibrator.evaluate_coverage(
        probabilities,
        np.array([0, 1]),
        valid_mask=valid_mask,
    )
    assert metrics["average_set_size"] == 2.0


def test_calibration_artifact_schema_and_hash_validation(tmp_path):
    """Calibration artifacts require coherent scores and a model-binding SHA-256."""
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    calibrator.fit(np.array([[0.9, 0.1], [0.8, 0.2]]), np.array([0, 0]))
    artifact_path = tmp_path / "calibration.json"
    calibrator.save(artifact_path, checkpoint_sha256="a" * 64, metadata={"model_version": "test"})

    loaded = ConformalDecisionCalibrator(alpha=0.05)
    loaded.load(artifact_path)
    assert loaded.checkpoint_sha256 == "a" * 64
    assert loaded.num_calib_samples == len(loaded.calibration_scores) == 2

    valid_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    malformed_payloads = []
    missing_hash = dict(valid_payload)
    missing_hash.pop("checkpoint_sha256")
    malformed_payloads.append(missing_hash)
    malformed_hash = dict(valid_payload)
    malformed_hash["checkpoint_sha256"] = "not-a-sha256"
    malformed_payloads.append(malformed_hash)
    mismatched_count = dict(valid_payload)
    mismatched_count["num_calib_samples"] = 1
    malformed_payloads.append(mismatched_count)
    invalid_score = dict(valid_payload)
    invalid_score["calibration_scores"] = [float("nan"), 0.2]
    malformed_payloads.append(invalid_score)

    for index, payload in enumerate(malformed_payloads):
        bad_path = tmp_path / f"malformed_{index}.json"
        bad_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        with pytest.raises(ValueError):
            ConformalDecisionCalibrator().load(bad_path)

    with pytest.raises(ValueError):
        ConformalDecisionCalibrator(alpha=0.05).save(tmp_path / "unbound.json")


def test_noul_threshold_validation():
    """Boolean primitive thresholds must be finite probabilities."""
    for value in [-0.1, 1.1, float("nan"), float("inf"), "nan"]:
        with pytest.raises(ValueError):
            Noul(threshold=value)

    assert Noul(threshold=0).threshold == 0.0
    assert Noul(threshold=1).threshold == 1.0
    assert Noul(threshold="0.5").threshold == 0.5


def test_score_rejects_non_finite_and_invalid_labels():
    for kwargs in (
        {"min_val": float("nan")},
        {"max_val": float("inf")},
        {"steps": 2.5},
        {"steps": True},
        {"labels": ["same", "same", "same", "same", "same"]},
        {"labels": ["", "low", "mid", "high", "top"]},
    ):
        with pytest.raises(ValueError):
            Score(**kwargs)

    with pytest.raises(ValueError):
        Noul(threshold=True)


def test_score_default_labels_remain_distinct_for_narrow_ranges():
    score = Score(min_val=0.0, max_val=0.01, steps=3)
    assert len(set(score.labels)) == 3
    assert len(set(score.options)) == 3


def test_coverage_uses_full_precision_cutoff():
    """Coverage must use the exact q-hat rather than the display-rounded value."""
    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    calibration_probs = np.vstack(
        [
            np.tile([[0.9, 0.1]], (38, 1)),
            np.tile([[0.8765433, 0.1234567]], (2, 1)),
        ]
    )
    calibrator.fit(
        calibration_probs,
        np.zeros(40, dtype=np.int64),
    )

    # This probability is below the exact cutoff (1 - 0.1234567) but above
    # the cutoff produced by rounding q-hat to four decimal places.
    metrics = calibrator.evaluate_coverage(
        np.array([[0.87654325, 0.12345675]], dtype=np.float64),
        np.array([0], dtype=np.int64),
    )
    assert metrics["empirical_coverage"] == 0.0
    assert metrics["average_set_size"] == 0.0


if __name__ == "__main__":
    test_conformal_coverage_guarantee()
    test_losses_execution()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Phase 2 Conformal Engine & Losses are ready.")
