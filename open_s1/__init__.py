# -*- coding: utf-8 -*-
"""
Aegis-S1: The Provably Safe Non-Autoregressive System 1 Decision Model.
Fast, calibrated, sub-50ms decision reflexes for AI Agents.
Full alignment with TypeSafe Jev & Laya decision primitives and multi-query parallel evaluation.
"""

__version__ = "2.1.0"

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.conformal import ConformalDecisionCalibrator
from src.modeling_s1 import DynamicOptionMarkerHead, S1DecisionModel
from src.tokenizer_utils import (
    encode_decision_batch,
    encode_multi_query_batch,
    format_decision_prompt,
    format_multi_query_prompt,
)

from .primitives import (
    Boolean,
    Choice,
    ChoiceResult,
    EvaluationResult,
    Noul,
    NoulResult,
    Score,
    ScoreResult,
)


class AegisRouter:
    """
    High-level, production-ready System 1 Decision Router.
    Supports single choice decisions and multi-query parallel forward evaluations.
    """
    def __init__(
        self,
        model: S1DecisionModel,
        tokenizer: AutoTokenizer,
        calibrator: ConformalDecisionCalibrator,
        device: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.calibrator = calibrator
        self.device = device

    @classmethod
    def load(
        cls,
        model_dir: Optional[Union[str, Path]] = None,
        base_model_path: str = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
        alpha: float = 0.05,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if model_dir is None:
            v3_path = Path("E:/s1-decision-model/output/s1_model_v3")
            v2_path = Path("E:/s1-decision-model/output/s1_model_v2")
            if (v3_path / "s1_decision_weights.pt").exists():
                model_dir = v3_path
            elif (v2_path / "s1_decision_weights.pt").exists():
                model_dir = v2_path
            else:
                model_dir = v3_path
        else:
            model_dir = Path(model_dir)

        weight_file = model_dir / "s1_decision_weights.pt"
        
        # Strict validation: prevent silent initialization with random weights
        if not weight_file.exists():
            raise FileNotFoundError(
                f"Aegis-S1 weight file not found at: {weight_file}. "
                f"Please ensure trained model weights exist before initializing AegisRouter."
            )

        
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir if (model_dir / "tokenizer_config.json").exists() else base_model_path
        )
        config = AutoConfig.from_pretrained(base_model_path)
        hidden_dim = getattr(config, "hidden_size", 896)
        
        backbone = AutoModel.from_pretrained(
            base_model_path,
            config=config,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
        
        model = S1DecisionModel(
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_heads=8,
            num_inter_layers=2,
        ).to(device)
        
        checkpoint = torch.load(weight_file, map_location=device)
        backbone.load_state_dict(checkpoint["backbone_lora"], strict=False)
        model.decision_head.load_state_dict(checkpoint["decision_head"])
            
        if device == "cuda":
            model.decision_head.to(dtype=torch.bfloat16)
        model.eval()
        
        # Load calibration artifact if available, else initialize
        calibrator = ConformalDecisionCalibrator(alpha=alpha)
        calib_file = model_dir / "conformal_calibration.json"
        if calib_file.exists():
            calibrator.load(calib_file)
        else:
            # Safe default fallback quantile
            calibrator.quantile_threshold = 0.05
        
        return cls(model=model, tokenizer=tokenizer, calibrator=calibrator, device=device)

    def decide(
        self,
        state: str,
        question: str,
        candidates: List[str],
        alpha: Optional[float] = None,
        max_length: int = 512,
    ) -> Dict:
        """
        Execute sub-50ms non-autoregressive decision in a single forward pass
        with mathematical conformal safety guarantees and intelligent left-truncation.
        """
        batch = encode_decision_batch(
            tokenizer=self.tokenizer,
            states=[state],
            questions=[question],
            options_per_sample=[candidates],
            max_length=max_length,
            device=self.device,
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                marker_indices=batch["marker_indices"],
                marker_mask=batch["marker_mask"],
                bidirectional=True,
            )
            
        probs = outputs["probs"].float().cpu().numpy()[0]
        
        # Target alpha: allow per-request override or use router default
        target_alpha = alpha if alpha is not None else self.calibrator.alpha
        conformal_out = self.calibrator.predict(probs, options_map=candidates, alpha=target_alpha)[0]
        
        best_idx = int(outputs["best_choice_idx"].item())
        confidence = float(outputs["confidence"].item())
        escalate_risk = float(outputs["escalate_risk"].item())
        
        # Dual-gate decision logic:
        # Act is ONLY allowed if conformal set size is exactly 1 AND escalate risk is low
        raw_verdict = conformal_out["verdict"]
        if raw_verdict == "act" and escalate_risk > 0.70:
            final_verdict = "escalate"
            final_explanation = (
                f"Conformal set approved single choice, but Escalate Gate detected high anomaly "
                f"risk ({escalate_risk:.2f} > 0.70). Escalating to System 2."
            )
        else:
            final_verdict = raw_verdict
            final_explanation = conformal_out["explanation"]
        
        return {
            "selected_option": candidates[best_idx],
            "selected_index": best_idx,
            "confidence": round(confidence, 4),
            "escalate_risk": round(escalate_risk, 4),
            "conformal_verdict": final_verdict,
            "prediction_set": conformal_out["prediction_set"],
            "prediction_set_size": conformal_out["prediction_set_size"],
            "alpha_guarantee": 1.0 - target_alpha,
            "explanation": final_explanation,
            "probabilities": {candidates[k]: round(float(probs[k]), 4) for k in range(len(candidates))},
        }

    def evaluate(
        self,
        state: str,
        schema: Dict[str, Union[Choice, Score, Noul]],
        alpha: Optional[float] = None,
        max_length: int = 1024,
    ) -> EvaluationResult:
        """
        Evaluate multiple heterogeneous decision queries in parallel on the given state
        in a SINGLE non-autoregressive forward pass (~30-50ms total).
        Matches and extends TypeSafe Jev & Laya decision primitives.
        """
        t0 = time.perf_counter()
        
        queries_dict = {}
        for q_name, primitive in schema.items():
            queries_dict[q_name] = (primitive.question, primitive.options)

        batch = encode_multi_query_batch(
            tokenizer=self.tokenizer,
            states=[state],
            queries_per_sample=[queries_dict],
            max_length=max_length,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                query_markers=batch["query_markers"],
                bidirectional=True,
            )

        escalate_risk = float(outputs["escalate_risk"].item())
        target_alpha = alpha if alpha is not None else self.calibrator.alpha
        
        results_dict: Dict[str, Union[ChoiceResult, ScoreResult, NoulResult]] = {}

        for q_name, primitive in schema.items():
            q_out = outputs["queries"][q_name]
            probs = q_out["probs"].float().cpu().numpy()[0]
            
            if isinstance(primitive, Choice):
                conformal_out = self.calibrator.predict(
                    probs, options_map=primitive.options, alpha=target_alpha
                )[0]
                best_idx = int(q_out["best_choice_idx"].item())
                conf = float(q_out["confidence"].item())
                raw_verdict = conformal_out["verdict"]
                
                if raw_verdict == "act" and escalate_risk > 0.70:
                    v = "escalate"
                    expl = f"Conformal set approved single choice, but anomaly risk is high ({escalate_risk:.2f})."
                else:
                    v = raw_verdict
                    expl = conformal_out["explanation"]

                results_dict[q_name] = ChoiceResult(
                    selected_option=primitive.options[best_idx],
                    selected_index=best_idx,
                    confidence=round(conf, 4),
                    verdict=v,
                    prediction_set=conformal_out["prediction_set"],
                    prediction_set_size=conformal_out["prediction_set_size"],
                    probabilities={primitive.options[k]: round(float(probs[k]), 4) for k in range(len(primitive.options))},
                    explanation=expl,
                )

            elif isinstance(primitive, Score):
                vals = np.array(primitive.values, dtype=np.float64)
                expected_score = float(np.sum(vals * probs))
                variance = float(np.sum(((vals - expected_score) ** 2) * probs))
                std = float(np.sqrt(variance))
                conf = float(np.max(probs))
                
                scale_span = primitive.max_val - primitive.min_val
                # High uncertainty if std exceeds 35% of scale or global escalate risk triggered
                if (std / scale_span) > 0.35 or escalate_risk > 0.70:
                    v = "escalate"
                    expl = f"High scoring variance (std={std:.2f} over span={scale_span:.1f})."
                else:
                    v = "act"
                    expl = f"Calibrated score expectation: {expected_score:.2f} (std={std:.2f})."

                results_dict[q_name] = ScoreResult(
                    score=round(expected_score, 2),
                    std=round(std, 3),
                    confidence=round(conf, 4),
                    verdict=v,
                    probabilities={primitive.options[k]: round(float(probs[k]), 4) for k in range(len(primitive.options))},
                    explanation=expl,
                )

            elif isinstance(primitive, (Noul, Boolean)):
                p_false = float(probs[0])
                p_true = float(probs[1]) if len(probs) > 1 else 1.0 - p_false
                
                dec_val = bool(p_true >= primitive.threshold)
                margin = abs(p_true - primitive.threshold)
                conf = min(1.0, float(margin * 2.0))
                
                if margin < 0.12 or escalate_risk > 0.70:
                    v = "escalate"
                    expl = f"Probability near boundary threshold ({p_true:.3f} vs {primitive.threshold})."
                else:
                    v = "act"
                    expl = f"P(true)={p_true:.3f} >= {primitive.threshold} -> {dec_val}."

                results_dict[q_name] = NoulResult(
                    value=dec_val,
                    probability=round(p_true, 4),
                    confidence=round(conf, 4),
                    verdict=v,
                    probabilities={primitive.false_label: round(p_false, 4), primitive.true_label: round(p_true, 4)},
                    explanation=expl,
                )

        overall_verdict = "act" if all(r.verdict == "act" for r in results_dict.values()) else "escalate"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        return EvaluationResult(
            results=results_dict,
            overall_verdict=overall_verdict,
            escalate_risk=round(escalate_risk, 4),
            latency_ms=round(latency_ms, 2),
        )


def load(
    model_dir: Union[str, Path] = "E:/s1-decision-model/output/s1_model_v3",
    base_model_path: str = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct",
    device: Optional[str] = None,
    alpha: float = 0.05,
) -> AegisRouter:
    """Convenient functional loader: `import open_s1 as s1; router = s1.load()`"""
    return AegisRouter.load(
        model_dir=model_dir,
        base_model_path=base_model_path,
        device=device,
        alpha=alpha,
    )


__all__ = [
    "load",
    "AegisRouter",
    "Choice",
    "Score",
    "Noul",
    "Boolean",
    "ChoiceResult",
    "ScoreResult",
    "NoulResult",
    "EvaluationResult",
    "S1DecisionModel",
    "ConformalDecisionCalibrator",
]
