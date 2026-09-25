# -*- coding: utf-8 -*-
"""
Aegis-S1: The Provably Safe Non-Autoregressive System 1 Decision Model.
Fast, calibrated, sub-50ms decision reflexes for AI Agents.
Full alignment with TypeSafe Jev & Laya decision primitives and multi-query parallel evaluation.
"""

__version__ = "2.1.0"

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

logger = logging.getLogger("open_s1")
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.conformal import ConformalDecisionCalibrator
from src.modeling_s1 import DynamicOptionMarkerHead, S1DecisionModel
from src.provenance import (
    resolve_provenance_file,
    sha256_file,
    sha256_model_directory,
    sha256_tokenizer,
    validate_provenance_pair,
)
from src.tokenizer_utils import (
    encode_decision_batch,
    encode_multi_query_batch,
    format_decision_prompt,
    format_multi_query_prompt,
    load_tokenizer_checked,
    MAX_CANDIDATE_COUNT,
    MAX_CANDIDATE_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SEQUENCE_LENGTH,
    MAX_STATE_CHARS,
    normalize_candidates,
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


def _validate_alpha(alpha: float, name: str = "alpha") -> float:
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float strictly between 0.0 and 1.0") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be a finite float strictly between 0.0 and 1.0, got {alpha!r}")
    return value


def _validate_probability_vector(probs: np.ndarray, name: str = "probabilities") -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise RuntimeError(f"{name} must be a one-dimensional vector with at least two entries")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise RuntimeError(f"{name} contains non-finite or out-of-range values")
    if not np.isclose(float(values.sum()), 1.0, atol=1e-2, rtol=0.0):
        raise RuntimeError(f"{name} must sum to 1.0, got {float(values.sum()):.8f}")
    return values


def _validate_gate_value(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"model output {name} must be finite and within [0, 1], got {value!r}")
    return value


def _tokenizer_sha256(tokenizer) -> str:
    return sha256_tokenizer(tokenizer)


def _validate_artifact_provenance(
    calibrator: ConformalDecisionCalibrator,
    checkpoint: Dict[str, Any],
    tokenizer_sha256: Optional[str] = None,
    base_model_sha256: Optional[str] = None,
    base_model_path: Optional[Path] = None,
    root_dir: Optional[Path] = None,
):
    """Reject legacy or incomplete artifacts before they can drive autonomous decisions."""
    metadata = getattr(calibrator, "metadata", None)
    validate_provenance_pair(checkpoint, metadata)
    config = checkpoint["config"]
    # Paths are descriptive and may change when a checkpoint moves between hosts;
    # the content hash below is the model identity check.
    if tokenizer_sha256 is not None and tokenizer_sha256.casefold() != str(metadata["tokenizer_sha256"]).casefold():
        raise RuntimeError(
            "Tokenizer hash mismatch between calibration metadata and the tokenizer loaded from model_dir"
        )
    if base_model_sha256 is not None and base_model_sha256.casefold() != str(metadata["base_model_sha256"]).casefold():
        raise RuntimeError(
            "Base model hash mismatch between calibration metadata and the selected base model files"
        )

    root = Path(root_dir or Path(__file__).resolve().parent.parent)
    for path_field, hash_field in (
        ("calibrated_split", "calibration_data_sha256"),
        ("train_split", "train_data_sha256"),
        ("test_split", "test_data_sha256"),
    ):
        data_path = resolve_provenance_file(metadata[path_field], root)
        actual_hash = sha256_file(data_path)
        if actual_hash.casefold() != str(metadata[hash_field]).casefold():
            raise RuntimeError(
                f"Provenance data hash mismatch for {path_field}: "
                f"declared={metadata[hash_field]}, actual={actual_hash}"
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
        base_model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        alpha: float = 0.05,
        fuse_lora: bool = True,
        provenance_root: Optional[Union[str, Path]] = None,
    ):
        alpha = _validate_alpha(alpha)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if base_model_path is None:
            base_model_path = os.environ.get(
                "BASE_MODEL_PATH", "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
            )
        base_model_path = Path(base_model_path).expanduser().resolve()
        if not base_model_path.exists():
            raise FileNotFoundError(
                f"Base model path not found: {base_model_path}. Set BASE_MODEL_PATH or pass base_model_path explicitly."
            )
        base_model_sha256 = sha256_model_directory(base_model_path)

        if model_dir is None:
            repo_root = Path(__file__).resolve().parent.parent
            default_v6_dir = repo_root / "output" / "s1_model_v6"
            if not (default_v6_dir / "s1_decision_weights.pt").exists():
                try:
                    from scripts.download_v6_checkpoint import download_from_hf
                    logger.info("Local V6 weights missing. Triggering automatic download from Hugging Face...")
                    download_from_hf(default_v6_dir)
                except Exception as dl_err:
                    logger.warning(f"Could not automatically download weights: {dl_err}")
            model_dir = default_v6_dir
        else:
            model_dir = Path(model_dir)

        weight_file = model_dir / "s1_decision_weights.pt"
        
        if not weight_file.exists():
            raise FileNotFoundError(
                f"Millennium-Jev model weights not found at: {weight_file}.\n"
                f"Please run 'python scripts/download_v6_checkpoint.py' to automatically download and verify "
                f"the official V6 release (SHA-256: eaf07edd808cecf43473a13e6a331f57bc8d81696f2a1ed7d82dbc7467e01191).\n"
                f"⚠️ Note: Do NOT use legacy V4 checkpoints as they are officially deprecated."
            )

        checkpoint = torch.load(weight_file, map_location=device)
        if not isinstance(checkpoint, dict) or not {"backbone_lora", "decision_head"}.issubset(checkpoint):
            raise RuntimeError("Checkpoint must contain backbone_lora and decision_head state dictionaries")

        calibrator = ConformalDecisionCalibrator(alpha=alpha)
        calib_file = model_dir / "conformal_calibration.json"
        if not calib_file.exists():
            raise FileNotFoundError(
                f"Validated calibration artifact not found at: {calib_file}. "
                "Refusing to run autonomous decisions without calibration."
            )
        try:
            calibrator.load(calib_file, alpha=alpha)
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Invalid or legacy calibration artifact '{calib_file}': {exc}. "
                "Retrain and recalibrate on the current disjoint_v6 splits."
            ) from exc
        checkpoint_sha256 = getattr(calibrator, "checkpoint_sha256", None)
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise RuntimeError(
                f"Calibration artifact '{calib_file.name}' must contain a 64-character checkpoint_sha256 binding"
            )
        actual_sha256 = sha256_file(weight_file)
        if actual_sha256 != checkpoint_sha256.lower():
            raise RuntimeError(
                f"Cryptographic hash mismatch! The calibration file '{calib_file.name}' "
                f"was fitted on checkpoint SHA-256 '{checkpoint_sha256}', "
                f"but loaded checkpoint '{weight_file.name}' has SHA-256 '{actual_sha256}'. "
                "Refusing to apply invalid conformal risk parameters."
            )
        if not calibrator.calibration_scores or calibrator.num_calib_samples <= 0:
            raise RuntimeError("Calibration artifact must contain non-empty calibration scores")
        tokenizer_path = model_dir if (model_dir / "tokenizer_config.json").exists() else base_model_path
        tokenizer = load_tokenizer_checked(tokenizer_path)
        _validate_artifact_provenance(
            calibrator,
            checkpoint,
            _tokenizer_sha256(tokenizer),
            base_model_sha256=base_model_sha256,
            base_model_path=base_model_path,
            root_dir=(
                Path(provenance_root).expanduser().resolve()
                if provenance_root is not None
                else (
                    Path(os.environ["S1_PROVENANCE_ROOT"]).expanduser().resolve()
                    if os.environ.get("S1_PROVENANCE_ROOT")
                    else None
                )
            ),
        )
        config = AutoConfig.from_pretrained(base_model_path)
        hidden_dim = getattr(config, "hidden_size", 896)
        
        backbone = AutoModel.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        lora_state = checkpoint["backbone_lora"]
        if not isinstance(lora_state, dict) or not lora_state:
            raise RuntimeError("Checkpoint contains an empty or invalid LoRA state dictionary")
        
        # Dynamically detect LoRA rank and target modules from checkpoint
        sample_lora_a = next((v for k, v in lora_state.items() if "lora_A" in k), None)
        detected_r = sample_lora_a.shape[0] if sample_lora_a is not None else 32
        
        detected_targets = set()
        for k in lora_state.keys():
            for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                if m in k:
                    detected_targets.add(m)
        required_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        missing_targets = sorted(required_targets - detected_targets)
        if missing_targets:
            raise RuntimeError(
                "Checkpoint LoRA state is missing required target modules: "
                + ", ".join(missing_targets)
            )
        target_modules = sorted(detected_targets)
        
        lora_config = LoraConfig(
            r=detected_r,
            lora_alpha=detected_r * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
        backbone_state = backbone.state_dict()
        invalid_lora_keys = []
        for key, value in lora_state.items():
            if key not in backbone_state or not isinstance(value, torch.Tensor):
                invalid_lora_keys.append(key)
            elif tuple(value.shape) != tuple(backbone_state[key].shape):
                invalid_lora_keys.append(key)
        if invalid_lora_keys:
            raise RuntimeError(
                f"Checkpoint LoRA state does not match the selected base model; invalid keys: {invalid_lora_keys[:5]}"
            )
        required_lora_keys = {key for key in backbone_state if "lora_" in key}
        absent_lora_keys = required_lora_keys - set(lora_state)
        if absent_lora_keys:
            raise RuntimeError(f"Checkpoint is missing LoRA keys: {sorted(absent_lora_keys)[:5]}")
        _, unexpected_keys = backbone.load_state_dict(lora_state, strict=False)
        if unexpected_keys:
            raise RuntimeError(f"Checkpoint contains unexpected LoRA keys: {unexpected_keys[:5]}")
        
        if fuse_lora and hasattr(backbone, "merge_and_unload"):
            backbone = backbone.merge_and_unload()
            logger.info("[*] Fused LoRA adapter into base weights for 2x faster forward inference.")

        model = S1DecisionModel(
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_heads=8,
            num_inter_layers=2,
        ).to(device)

        model.decision_head.load_state_dict(checkpoint["decision_head"])

        if device == "cuda":
            model.decision_head.to(dtype=torch.bfloat16)
        model.eval()
        model.enable_fast_bidirectional(True)
        
        return cls(model=model, tokenizer=tokenizer, calibrator=calibrator, device=device)

    def decide(
        self,
        state: str,
        question: str,
        candidates: List[str],
        alpha: Optional[float] = None,
        max_length: int = 2048,
    ) -> Dict:
        if not isinstance(state, str) or len(state) > MAX_STATE_CHARS:
            raise ValueError(f"state must be a string of at most {MAX_STATE_CHARS} characters")
        if not isinstance(question, str) or len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be a string of at most {MAX_QUESTION_CHARS} characters")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= MAX_SEQUENCE_LENGTH:
            raise ValueError(f"max_length must be an integer between 1 and {MAX_SEQUENCE_LENGTH}")
        candidates = normalize_candidates(candidates)

        if alpha is not None:
            target_alpha = _validate_alpha(alpha)
        else:
            target_alpha = _validate_alpha(self.calibrator.alpha, "calibrator.alpha")

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
                option_spans=batch.get("option_spans"),
                bidirectional=True,
            )
            
        probs = _validate_probability_vector(outputs["probs"].float().cpu().numpy()[0])
        
        # Target alpha: allow per-request override or use router default
        conformal_out = self.calibrator.predict(probs, options_map=candidates, alpha=target_alpha)[0]
        
        best_idx = int(outputs["best_choice_idx"].item())
        if best_idx < 0 or best_idx >= len(candidates) or best_idx != int(np.argmax(probs)):
            raise RuntimeError("model best_choice_idx is inconsistent with its probability vector")
        confidence = _validate_gate_value(outputs["confidence"].item(), "confidence")
        escalate_risk = _validate_gate_value(outputs["escalate_risk"].item(), "escalate_risk")
        
        # Tri-gate decision logic:
        # Act is ONLY allowed if:
        # 1. Conformal set size is exactly 1 (raw_verdict == 'act')
        # 2. Top choice confidence >= 0.60 (prevents flat-distribution OOD gibberish from acting)
        # 3. Escalate Gate anomaly risk <= 0.70
        raw_verdict = conformal_out["verdict"]
        if raw_verdict == "act" and escalate_risk > 0.70:
            final_verdict = "escalate"
            final_explanation = (
                f"Conformal set approved single choice, but Escalate Gate detected high anomaly "
                f"risk ({escalate_risk:.2f} > 0.70). Escalating to System 2."
            )
        elif raw_verdict == "act" and confidence < 0.60:
            final_verdict = "escalate"
            final_explanation = (
                f"Conformal set approved single choice, but Top-1 confidence is below safety boundary "
                f"({confidence:.2f} < 0.60). Escalating to System 2 to prevent OOD hallucination."
            )
        else:
            final_verdict = raw_verdict
            final_explanation = conformal_out["explanation"]
        
        return {
            "can_act": (final_verdict == "act"),
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
        max_length: int = 2048,
    ) -> EvaluationResult:
        """
        Evaluate multiple heterogeneous decision queries in parallel on the given state
        in a SINGLE non-autoregressive forward pass (~30-50ms total).
        Matches and extends TypeSafe Jev & Laya decision primitives.
        """
        t0 = time.perf_counter()
        
        if not schema:
            raise ValueError("Schema dictionary cannot be empty. Must provide at least one decision primitive.")
        if not isinstance(state, str) or len(state) > MAX_STATE_CHARS:
            raise ValueError(f"state must be a string of at most {MAX_STATE_CHARS} characters")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= MAX_SEQUENCE_LENGTH:
            raise ValueError(f"max_length must be an integer between 1 and {MAX_SEQUENCE_LENGTH}")
        if alpha is not None:
            target_alpha = _validate_alpha(alpha)
        else:
            target_alpha = _validate_alpha(self.calibrator.alpha, "calibrator.alpha")

        queries_dict = {}
        for q_name, primitive in schema.items():
            if not isinstance(q_name, str) or not q_name.strip():
                raise ValueError("schema query names must be non-empty strings")
            if not isinstance(primitive, (Choice, Score, Noul, Boolean)):
                raise TypeError(
                    f"schema query '{q_name}' must be a Choice, Score, Noul, or Boolean primitive"
                )
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

        escalate_risk = _validate_gate_value(outputs["escalate_risk"].item(), "escalate_risk")
        target_alpha = _validate_alpha(alpha, "alpha") if alpha is not None else _validate_alpha(self.calibrator.alpha, "calibrator.alpha")
        
        results_dict: Dict[str, Union[ChoiceResult, ScoreResult, NoulResult]] = {}

        for q_name, primitive in schema.items():
            q_out = outputs["queries"][q_name]
            probs = _validate_probability_vector(q_out["probs"].float().cpu().numpy()[0], f"probabilities[{q_name}]")
            conformal_out = self.calibrator.predict(
                probs, options_map=primitive.options, alpha=target_alpha
            )[0]
            raw_verdict = conformal_out["verdict"]
            best_idx = int(q_out["best_choice_idx"].item())
            if best_idx < 0 or best_idx >= len(primitive.options) or best_idx != int(np.argmax(probs)):
                raise RuntimeError(f"model best_choice_idx is inconsistent for query '{q_name}'")
            conf = _validate_gate_value(q_out["confidence"].item(), f"confidence[{q_name}]")

            def tri_gate_verdict(explanation: str):
                if raw_verdict == "act" and escalate_risk > 0.70:
                    return "escalate", f"{explanation} Escalate Gate risk is high ({escalate_risk:.2f} > 0.70)."
                if raw_verdict == "act" and conf < 0.60:
                    return "escalate", f"{explanation} Top-1 confidence is below 0.60 ({conf:.2f})."
                return raw_verdict, explanation
            
            if isinstance(primitive, Choice):
                v, expl = tri_gate_verdict(conformal_out["explanation"])

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
                if (std / scale_span) > 0.35:
                    v = "escalate"
                    expl = f"High scoring variance (std={std:.2f} over span={scale_span:.1f})."
                else:
                    expl = f"Calibrated score expectation: {expected_score:.2f} (std={std:.2f})."
                    v, expl = tri_gate_verdict(expl)

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
                
                if margin < 0.12:
                    v = "escalate"
                    expl = f"Probability near boundary threshold ({p_true:.3f} vs {primitive.threshold})."
                else:
                    expl = f"P(true)={p_true:.3f} >= {primitive.threshold} -> {dec_val}."
                    v, expl = tri_gate_verdict(expl)

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
    model_dir: Optional[Union[str, Path]] = None,
    base_model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    alpha: float = 0.05,
    fuse_lora: bool = True,
    provenance_root: Optional[Union[str, Path]] = None,
) -> AegisRouter:
    """Convenient functional loader: `import open_s1 as s1; router = s1.load()`"""
    return AegisRouter.load(
        model_dir=model_dir,
        base_model_path=base_model_path,
        device=device,
        alpha=alpha,
        fuse_lora=fuse_lora,
        provenance_root=provenance_root,
    )


MillenniumRouter = AegisRouter

__all__ = [
    "load",
    "MillenniumRouter",
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
