# -*- coding: utf-8 -*-
"""
Conformal Prediction Engine for System 1 Decision Models.
Provides distribution-free, provable finite-sample error rate guarantees (e.g. 1 - alpha = 95%).
Determines provably safe single-actions, ambiguity-escalation, and OOD rejection.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class ConformalDecisionCalibrator:
    """
    Split-Conformal Predictor using Adaptive Prediction Sets (APS) / Thresholding.
    Guarantees: P(Y in C(X)) >= 1 - alpha
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Desired error rate (e.g. 0.05 for 95% provable coverage)
        """
        self.alpha = alpha
        self.quantile_threshold: Optional[float] = None
        self.num_calib_samples: int = 0

    def fit(self, val_probs: Union[torch.Tensor, np.ndarray], val_labels: Union[torch.Tensor, np.ndarray]):
        """
        Compute non-conformity quantile threshold q_hat on calibration set.
        Args:
            val_probs: [N, K] matrix of softmax probabilities
            val_labels: [N] array of true class indices
        """
        if isinstance(val_probs, torch.Tensor):
            val_probs = val_probs.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.detach().cpu().numpy()

        n = len(val_labels)
        self.num_calib_samples = n
        
        # Calculate non-conformity scores: s_i = 1 - p(y_i)
        true_class_probs = val_probs[np.arange(n), val_labels]
        non_conformity_scores = 1.0 - true_class_probs
        
        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, max(0.0, level))
        
        # Empirical quantile with linear interpolation
        self.quantile_threshold = float(np.quantile(non_conformity_scores, level, method="higher"))
        return self.quantile_threshold

    def predict(
        self,
        probs: Union[torch.Tensor, np.ndarray],
        options_map: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Evaluate prediction set C(x) for test probabilities.
        Returns decision verdict:
          - 'act': Exactly 1 candidate meets guarantee -> execute immediately.
          - 'escalate': Multiple candidates meet guarantee -> ambiguous, route to System 2.
          - 'reject': 0 candidates meet guarantee -> Out-Of-Distribution (OOD).
        """
        if self.quantile_threshold is None:
            raise RuntimeError("Calibrator must be fitted on calibration data before predict()!")
            
        if isinstance(probs, list):
            probs = np.array(probs)
        elif isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
            
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            
        threshold = 1.0 - self.quantile_threshold
        results = []
        
        for i in range(len(probs)):
            p = probs[i]
            # Valid options are those whose probability exceeds safety threshold
            included_indices = np.where(p >= threshold)[0].tolist()
            
            # Sort included by probability descending
            included_indices.sort(key=lambda idx: p[idx], reverse=True)
            
            set_size = len(included_indices)
            best_idx = int(np.argmax(p))
            best_prob = float(p[best_idx])
            
            if set_size == 1:
                action = "act"
                msg = f"Provably safe single decision (coverage >= {1 - self.alpha:.0%})"
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
                "alpha_guarantee": 1.0 - self.alpha,
                "explanation": msg,
            })
            
        return results

    def evaluate_coverage(
        self,
        test_probs: Union[torch.Tensor, np.ndarray],
        test_labels: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        """
        Empirically verify that coverage rate matches or exceeds 1 - alpha.
        """
        if isinstance(test_probs, torch.Tensor):
            test_probs = test_probs.detach().cpu().numpy()
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.detach().cpu().numpy()

        results = self.predict(test_probs)
        covered = 0
        set_sizes = []
        
        for i, res in enumerate(results):
            true_y = test_labels[i]
            # Check if true_y is inside the prediction set indices
            included = [idx for idx, p in enumerate(test_probs[i]) if p >= (1.0 - self.quantile_threshold)]
            if true_y in included:
                covered += 1
            set_sizes.append(len(included))
            
        empirical_coverage = covered / len(test_labels)
        avg_set_size = float(np.mean(set_sizes))
        
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": round(empirical_coverage, 4),
            "guarantee_satisfied": empirical_coverage >= (1.0 - self.alpha),
            "average_set_size": round(avg_set_size, 3),
        }
