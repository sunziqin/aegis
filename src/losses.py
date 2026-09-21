# -*- coding: utf-8 -*-
"""
Calibrated Loss Functions for System 1 Decision Models.
Includes Multi-Class Brier Score Loss, Expected Calibration Error (ECE),
and Compound Decision Loss with Escalate Gate supervision.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassBrierLoss(nn.Module):
    """
    Strictly proper scoring rule: Brier Score.
    Penalizes deviations of probabilities from true one-hot outcomes.
    Loss = mean( sum_k (p_k - y_k)^2 )
    """
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            probs: [Batch, MaxK] (softmax probabilities)
            targets: [Batch] (long integer indices of true choice)
            mask: [Batch, MaxK] (bool mask of valid options)
        """
        batch_size, max_k = probs.shape
        # Create one-hot targets
        one_hot = F.one_hot(targets, num_classes=max_k).to(dtype=probs.dtype)
        
        diff_sq = (probs - one_hot) ** 2
        if mask is not None:
            diff_sq = diff_sq * mask.to(dtype=probs.dtype)
            loss_per_sample = diff_sq.sum(dim=-1) / (mask.sum(dim=-1).clamp(min=1).to(dtype=probs.dtype))
        else:
            loss_per_sample = diff_sq.sum(dim=-1)
            
        return loss_per_sample.mean()


class ExpectedCalibrationError(nn.Module):
    """
    Computes Expected Calibration Error (ECE) across M confidence bins.
    """
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = (predictions == targets).float()
        
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device)
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece = ece + torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece


class CalibratedDecisionLoss(nn.Module):
    """
    Compound loss function:
    Total Loss = CrossEntropy + lambda_brier * BrierLoss + lambda_margin * MarginLoss + lambda_esc * EscalateLoss
    """
    def __init__(self, lambda_brier: float = 0.5, lambda_esc: float = 0.3, lambda_margin: float = 0.3, margin: float = 1.0):
        super().__init__()
        self.lambda_brier = lambda_brier
        self.lambda_esc = lambda_esc
        self.lambda_margin = lambda_margin
        self.margin = margin
        self.brier = MultiClassBrierLoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        escalate_risk: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits: [Batch, MaxK] (masked logits)
            probs: [Batch, MaxK]
            escalate_risk: [Batch] (predicted probability of ambiguity/error)
            targets: [Batch] (true choice index)
            mask: [Batch, MaxK] (valid option mask)
        """
        # 1. Standard Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # 2. Proper Scoring Rule: Brier Loss
        brier_loss = self.brier(probs, targets, mask)
        
        # 3. Hard-Negative Margin Loss: Enforce geometric separation between target and closest rival
        target_logits = logits.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        neg_logits = logits.clone()
        neg_logits.scatter_(1, targets.unsqueeze(1), -1e4)
        if mask is not None:
            neg_logits = neg_logits.masked_fill(~mask, -1e4)
        hardest_neg_logits = neg_logits.max(dim=1).values
        margin_loss = F.relu(self.margin - (target_logits - hardest_neg_logits)).mean()

        # 4. Escalate Risk Supervision:
        with torch.no_grad():
            preds = torch.argmax(probs, dim=-1)
            is_error = (preds != targets).float()
            is_low_conf = (torch.max(probs, dim=-1).values < 0.6).float()
            escalate_target = torch.clamp(is_error + is_low_conf, max=1.0).to(dtype=escalate_risk.dtype)
            
        esc_loss = self.bce(escalate_risk, escalate_target)
        
        total_loss = ce_loss + self.lambda_brier * brier_loss + self.lambda_margin * margin_loss + self.lambda_esc * esc_loss
        
        return total_loss, {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "brier_loss": brier_loss.item(),
            "margin_loss": margin_loss.item(),
            "esc_loss": esc_loss.item(),
        }

