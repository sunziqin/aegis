# -*- coding: utf-8 -*-
"""
Conformal Prediction Engine for System 1 Decision Models.
Provides distribution-free, provable finite-sample error rate guarantees (e.g. 1 - alpha = 95%).
Determines provably safe single-actions, ambiguity-escalation, and OOD rejection.
Includes calibration artifact serialization and dynamic alpha adjustment.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


_PROBABILITY_SUM_TOLERANCE = 1e-2
_CHECKPOINT_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_alpha(alpha: float, *, field_name: str = "alpha") -> float:
    """Return a finite conformal error rate in the open interval (0, 1)."""
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite float strictly between 0.0 and 1.0, got {alpha!r}") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{field_name} must be a finite float strictly between 0.0 and 1.0, got {alpha!r}")
    return value


def _validate_checkpoint_sha256(value: object) -> str:
    """Validate and normalize the checkpoint binding stored in an artifact."""
    if not isinstance(value, str) or _CHECKPOINT_SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal SHA-256 string")
    return value.lower()


def _as_probability_matrix(
    probs: Union[torch.Tensor, np.ndarray, List[float]],
) -> np.ndarray:
    """Validate probability rows without changing the values used for calibration."""
    try:
        if isinstance(probs, torch.Tensor):
            # NumPy has no native bfloat16 support; cast before crossing the boundary.
            array = probs.detach().cpu().float().numpy().astype(np.float64, copy=False)
        else:
            array = np.asarray(probs, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probs must be a numeric probability vector or matrix") from exc

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError("probs must be a non-empty vector or a 2D matrix with at least two classes")
    if not np.all(np.isfinite(array)):
        raise ValueError("probs contains NaN or infinite values; refusing to make a decision")
    if np.any(array < 0.0):
        raise ValueError("probs contains negative values; refusing to make a decision")

    row_sums = array.sum(axis=1)
    if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise ValueError("probs rows must have a finite positive sum")
    if not np.all(np.isclose(row_sums, 1.0, atol=_PROBABILITY_SUM_TOLERANCE, rtol=0.0)):
        raise ValueError(
            "probs rows must sum to 1.0 within tolerance "
            f"({_PROBABILITY_SUM_TOLERANCE:g})"
        )
    return array


def _as_label_vector(labels: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Validate calibration/evaluation labels as integer class indices."""
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    try:
        array = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be a one-dimensional integer array") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional integer array")
    if not np.issubdtype(array.dtype, np.integer):
        try:
            numeric = array.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must contain integer class indices") from exc
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("labels must contain integer class indices")
        array = numeric.astype(np.int64)
    return array.astype(np.int64, copy=False)


def _as_valid_mask(valid_mask, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if valid_mask is None:
        return None
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()
    mask = np.asarray(valid_mask)
    if mask.shape != shape:
        raise ValueError(f"valid_mask shape {mask.shape} must match probability shape {shape}")
    if mask.dtype != np.bool_:
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("valid_mask must contain boolean values")
        mask = mask.astype(bool)
    if np.any(mask.sum(axis=1) < 2):
        raise ValueError("each valid_mask row must contain at least two candidate classes")
    return mask


def _quantile_from_scores(scores: List[float], alpha: float) -> float:
    alpha = _validate_alpha(alpha)
    if not scores:
        raise RuntimeError("No calibration scores available. Call fit() or load() first.")
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    # Probability nonconformity scores are bounded by 1. A rank of n + 1
    # therefore uses 1 as the conservative threshold and includes every class.
    q = 1.0 if rank > n else float(sorted(scores)[rank - 1])
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("calibration quantile must be a finite value in [0, 1]")
    return q


class ConformalDecisionCalibrator:
    """
    Split-Conformal Predictor using Non-conformity Score Calibration.
    Guarantees: P(Y in C(X)) >= 1 - alpha
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Desired error rate (e.g. 0.05 for 95% provable marginal coverage)
        """
        self.alpha = _validate_alpha(alpha)
        self.quantile_threshold: Optional[float] = None
        self.num_calib_samples: int = 0
        self.calibration_scores: List[float] = []
        self.checkpoint_sha256: Optional[str] = None
        self.metadata: Dict = {}

    def fit(self, val_probs: Union[torch.Tensor, np.ndarray], val_labels: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Compute non-conformity quantile threshold q_hat on calibration set.
        Args:
            val_probs: [N, K] matrix of softmax probabilities
            val_labels: [N] array of true class indices
        """
        self.alpha = _validate_alpha(self.alpha)
        val_probs = _as_probability_matrix(val_probs)
        val_labels = _as_label_vector(val_labels)
        n = len(val_labels)
        if len(val_probs) != n:
            raise ValueError(f"val_probs rows ({len(val_probs)}) must match labels ({n})")
        if np.any(val_labels < 0) or np.any(val_labels >= val_probs.shape[1]):
            raise ValueError("labels contain an out-of-range class index")
        self.num_calib_samples = n
        
        # Calculate non-conformity scores: s_i = 1 - p(y_i)
        true_class_probs = val_probs[np.arange(n), val_labels]
        non_conformity_scores = (1.0 - true_class_probs).tolist()
        self.calibration_scores = sorted(non_conformity_scores)
        
        # Compute threshold for current default alpha
        self.quantile_threshold = self._compute_quantile(self.alpha)
        return self.quantile_threshold

    def _compute_quantile(self, alpha: float) -> float:
        """Compute empirical conformal quantile from sorted calibration scores."""
        return _quantile_from_scores(self.calibration_scores, alpha)

    def _validate_calibration_state(self) -> float:
        """Return q_hat only for a fitted, internally consistent calibration set."""
        if not self.calibration_scores or self.num_calib_samples != len(self.calibration_scores):
            raise ValueError("calibration_scores must be non-empty and match num_calib_samples")
        try:
            scores = np.asarray(self.calibration_scores, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("calibration_scores must contain numeric values") from exc
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("calibration_scores must contain finite values in [0, 1]")
        return self._compute_quantile(self.alpha)

    def save(
        self,
        file_path: Union[str, Path],
        checkpoint_sha256: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Persist calibration scores and parameters to JSON with cryptographic hash binding."""
        self.alpha = _validate_alpha(self.alpha)
        expected_q = self._validate_calibration_state()
        if self.quantile_threshold is None or not np.isfinite(float(self.quantile_threshold)) or not 0.0 <= float(self.quantile_threshold) <= 1.0:
            raise ValueError("quantile_threshold must be a finite value in [0, 1]")
        if not np.isclose(float(self.quantile_threshold), expected_q, atol=1e-8, rtol=0.0):
            raise ValueError("quantile_threshold does not match calibration_scores and alpha")
        checkpoint_sha = checkpoint_sha256 or self.checkpoint_sha256
        if checkpoint_sha is not None:
            checkpoint_sha = _validate_checkpoint_sha256(checkpoint_sha)
        else:
            raise ValueError("checkpoint_sha256 is required to bind calibration to model weights")
        artifact_metadata = metadata if metadata is not None else self.metadata
        if not isinstance(artifact_metadata, dict):
            raise ValueError("metadata must be a JSON object")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "quantile_threshold": self.quantile_threshold,
            "num_calib_samples": self.num_calib_samples,
            "checkpoint_sha256": checkpoint_sha,
            "metadata": artifact_metadata,
            "calibration_scores": self.calibration_scores,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: Union[str, Path], alpha: Optional[float] = None):
        """Load calibration state from JSON, preserving caller-specified alpha if provided."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("calibration artifact must contain a JSON object")
        required = {"alpha", "quantile_threshold", "num_calib_samples", "calibration_scores", "checkpoint_sha256"}
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"calibration artifact missing required fields: {', '.join(missing)}")

        file_alpha = _validate_alpha(data["alpha"], field_name="calibration alpha")
        requested_alpha = _validate_alpha(alpha) if alpha is not None else None
        raw_count = data["num_calib_samples"]
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
            raise ValueError("num_calib_samples must be a positive integer")
        raw_scores = data["calibration_scores"]
        if not isinstance(raw_scores, list) or not raw_scores:
            raise ValueError("calibration_scores must be a non-empty JSON list")
        scores: List[float] = []
        for score in raw_scores:
            try:
                value = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError("calibration_scores must contain numeric values") from exc
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("calibration_scores must contain finite values in [0, 1]")
            scores.append(value)
        if raw_count != len(scores):
            raise ValueError(
                f"num_calib_samples ({raw_count}) does not match calibration_scores length ({len(scores)})"
            )
        try:
            stored_q = float(data["quantile_threshold"])
        except (TypeError, ValueError) as exc:
            raise ValueError("quantile_threshold must be numeric") from exc
        if not np.isfinite(stored_q) or not 0.0 <= stored_q <= 1.0:
            raise ValueError("quantile_threshold must be a finite value in [0, 1]")
        checkpoint_sha = _validate_checkpoint_sha256(data["checkpoint_sha256"])
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")

        sorted_scores = sorted(scores)
        file_q = _quantile_from_scores(sorted_scores, file_alpha)
        if not np.isclose(stored_q, file_q, atol=1e-8, rtol=0.0):
            raise ValueError("quantile_threshold does not match calibration_scores and calibration alpha")

        self.alpha = requested_alpha if requested_alpha is not None else file_alpha
        self.num_calib_samples = raw_count
        self.calibration_scores = sorted_scores
        self.checkpoint_sha256 = checkpoint_sha
        self.metadata = metadata
        self.quantile_threshold = _quantile_from_scores(sorted_scores, self.alpha)

    def predict(
        self,
        probs: Union[torch.Tensor, np.ndarray, List[float]],
        options_map: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        valid_mask=None,
    ) -> List[Dict]:
        """
        Evaluate prediction set C(x) for test probabilities.
        Returns decision verdict:
          - 'act': Exactly 1 candidate meets guarantee -> execute immediately.
          - 'escalate': Multiple candidates meet guarantee -> ambiguous, route to System 2.
          - 'reject': 0 candidates meet guarantee -> Out-Of-Distribution (OOD).
        """
        target_alpha = _validate_alpha(alpha if alpha is not None else self.alpha)
        if not self.calibration_scores:
            raise RuntimeError("Calibrator has no fitted calibration scores; refusing to make a decision")
        probs = _as_probability_matrix(probs)
        valid_mask = _as_valid_mask(valid_mask, probs.shape)
        if valid_mask is not None:
            valid_probs = np.where(valid_mask, probs, 0.0)
            if not np.all(np.isclose(valid_probs.sum(axis=1), 1.0, atol=_PROBABILITY_SUM_TOLERANCE, rtol=0.0)):
                raise ValueError("valid probability rows must sum to 1.0")
            if np.any(np.abs(probs[~valid_mask]) > _PROBABILITY_SUM_TOLERANCE):
                raise ValueError("invalid probability entries must be zero")
        if options_map is not None and len(options_map) != probs.shape[1]:
            raise ValueError(
                f"options_map length ({len(options_map)}) must match probability classes ({probs.shape[1]})"
            )
        q_thresh = self._compute_quantile(target_alpha)
        prob_threshold = max(0.0, 1.0 - q_thresh)
        results = []
        
        for i in range(len(probs)):
            p = probs[i]
            # Valid options are those whose probability exceeds safety threshold
            candidate_mask = valid_mask[i] if valid_mask is not None else np.ones(len(p), dtype=bool)
            included_indices = np.where(candidate_mask & (p >= prob_threshold))[0].tolist()
            included_indices.sort(key=lambda idx: p[idx], reverse=True)

            set_size = len(included_indices)
            best_idx = int(np.argmax(np.where(candidate_mask, p, -np.inf)))
            best_prob = float(p[best_idx])
            
            if set_size == 1:
                action = "act"
                msg = f"Provably safe single decision (coverage >= {1 - target_alpha:.0%})"
            elif set_size > 1:
                action = "escalate"
                msg = f"Ambiguity detected: {set_size} options satisfy safety boundary. Escalate to System 2."
            else:
                action = "reject"
                msg = "No option meets provable safety threshold (Potential Out-of-Distribution)."

            candidate_names = (
                [options_map[idx] for idx in included_indices]
                if options_map is not None
                else included_indices
            )
            
            results.append({
                "verdict": action,
                "best_choice_idx": best_idx,
                "best_confidence": round(best_prob, 4),
                "prediction_set_size": set_size,
                "prediction_set": candidate_names,
                "alpha_guarantee": 1.0 - target_alpha,
                "quantile_threshold": round(q_thresh, 4),
                "prob_cutoff": round(prob_threshold, 4),
                "explanation": msg,
            })
            
        return results

    def evaluate_coverage(
        self,
        test_probs: Union[torch.Tensor, np.ndarray],
        test_labels: Union[torch.Tensor, np.ndarray],
        alpha: Optional[float] = None,
        valid_mask=None,
    ) -> Dict[str, float]:
        """
        Empirically verify that coverage rate matches or exceeds 1 - alpha,
        and calculate selective risk on accepted single actions.
        """
        test_probs = _as_probability_matrix(test_probs)
        valid_mask = _as_valid_mask(valid_mask, test_probs.shape)
        test_labels = _as_label_vector(test_labels)
        if len(test_probs) != len(test_labels):
            raise ValueError(f"test_probs rows ({len(test_probs)}) must match labels ({len(test_labels)})")
        if np.any(test_labels < 0) or np.any(test_labels >= test_probs.shape[1]):
            raise ValueError("labels contain an out-of-range class index")

        target_alpha = _validate_alpha(alpha if alpha is not None else self.alpha)
        # Keep coverage accounting on the full-precision threshold. `predict`
        # rounds values for display, which must not move boundary examples
        # into or out of the prediction set.
        q_thresh = self._compute_quantile(target_alpha)
        cutoff = 1.0 - q_thresh
        results = self.predict(test_probs, alpha=target_alpha, valid_mask=valid_mask)
        
        covered = 0
        act_count = 0
        act_errors = 0
        set_sizes = []
        
        for i, res in enumerate(results):
            true_y = int(test_labels[i])
            candidate_mask = valid_mask[i] if valid_mask is not None else np.ones(test_probs.shape[1], dtype=bool)
            included = [idx for idx, p in enumerate(test_probs[i]) if candidate_mask[idx] and p >= cutoff]
            set_sizes.append(len(included))
            
            if true_y in included:
                covered += 1
                
            if res["verdict"] == "act":
                act_count += 1
                if res["best_choice_idx"] != true_y:
                    act_errors += 1
                    
        empirical_coverage = covered / len(test_labels)
        act_coverage = act_count / len(test_labels)
        selective_risk = (act_errors / max(1, act_count)) if act_count > 0 else 0.0
        
        return {
            "target_coverage": 1.0 - target_alpha,
            "empirical_coverage": round(empirical_coverage, 4),
            "guarantee_satisfied": empirical_coverage >= (1.0 - target_alpha),
            "act_coverage_rate": round(act_coverage, 4),
            "selective_risk_on_act": round(selective_risk, 4),
            "abstention_rate": round(1.0 - act_coverage, 4),
            "average_set_size": round(float(np.mean(set_sizes)), 3),
        }
