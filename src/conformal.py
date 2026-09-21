# -*- coding: utf-8 -*-
"""
Conformal Prediction Engine for System 1 Decision Models.
Provides distribution-free, provable finite-sample error rate guarantees (e.g. 1 - alpha = 95%).
Determines provably safe single-actions, ambiguity-escalation, and OOD rejection.
Includes calibration artifact serialization and dynamic alpha adjustment.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


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
        self.alpha = alpha
        self.quantile_threshold: Optional[float] = None
        self.num_calib_samples: int = 0
        self.calibration_scores: List[float] = []

    def fit(self, val_probs: Union[torch.Tensor, np.ndarray], val_labels: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Compute non-conformity quantile threshold q_hat on calibration set.
        Args:
            val_probs: [N, K] matrix of softmax probabilities
            val_labels: [N] array of true class indices
        """
        if isinstance(val_probs, torch.Tensor):
            val_probs = val_probs.detach().cpu().float().numpy()
        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.detach().cpu().numpy()

        n = len(val_labels)
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
        if not self.calibration_scores:
            raise RuntimeError("No calibration scores available. Call fit() or load() first.")
        n = len(self.calibration_scores)
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = min(1.0, max(0.0, level))
        q = float(np.quantile(self.calibration_scores, level, method="higher"))
        return q

    def save(self, file_path: Union[str, Path]):
        """Persist calibration scores and parameters to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "quantile_threshold": self.quantile_threshold,
            "num_calib_samples": self.num_calib_samples,
            "calibration_scores": self.calibration_scores,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: Union[str, Path]):
        """Load calibration state from JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.alpha = float(data.get("alpha", 0.05))
        self.num_calib_samples = int(data.get("num_calib_samples", 0))
        self.calibration_scores = [float(s) for s in data.get("calibration_scores", [])]
        self.quantile_threshold = float(data.get("quantile_threshold", self._compute_quantile(self.alpha)))

    def predict(
        self,
        probs: Union[torch.Tensor, np.ndarray, List[float]],
        options_map: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict]:
        """
        Evaluate prediction set C(x) for test probabilities.
        Returns decision verdict:
          - 'act': Exactly 1 candidate meets guarantee -> execute immediately.
          - 'escalate': Multiple candidates meet guarantee -> ambiguous, route to System 2.
          - 'reject': 0 candidates meet guarantee -> Out-Of-Distribution (OOD).
        """
        target_alpha = alpha if alpha is not None else self.alpha
        q_thresh = self._compute_quantile(target_alpha) if self.calibration_scores else (self.quantile_threshold or 0.05)
            
        if isinstance(probs, list):
            probs = np.array(probs)
        elif isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().float().numpy()
            
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            
        prob_threshold = max(0.0, 1.0 - q_thresh)
        results = []
        
        for i in range(len(probs)):
            p = probs[i]
            # Valid options are those whose probability exceeds safety threshold
            included_indices = np.where(p >= prob_threshold)[0].tolist()
            included_indices.sort(key=lambda idx: p[idx], reverse=True)
            
            set_size = len(included_indices)
            best_idx = int(np.argmax(p))
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
                if options_map and all(idx < len(options_map) for idx in included_indices)
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
    ) -> Dict[str, float]:
        """
        Empirically verify that coverage rate matches or exceeds 1 - alpha,
        and calculate selective risk on accepted single actions.
        """
        if isinstance(test_probs, torch.Tensor):
            test_probs = test_probs.detach().cpu().float().numpy()
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.detach().cpu().numpy()

        target_alpha = alpha if alpha is not None else self.alpha
        results = self.predict(test_probs, alpha=target_alpha)
        
        covered = 0
        act_count = 0
        act_errors = 0
        set_sizes = []
        
        for i, res in enumerate(results):
            true_y = int(test_labels[i])
            q_thresh = res["quantile_threshold"]
            cutoff = 1.0 - q_thresh
            
            included = [idx for idx, p in enumerate(test_probs[i]) if p >= cutoff]
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
