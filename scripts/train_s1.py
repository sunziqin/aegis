# -*- coding: utf-8 -*-
"""
Training Pipeline for Project Aegis-S1 (Non-Autoregressive Decision Model).
Trains LoRA on Qwen2.5-0.5B backbone + DynamicOptionMarkerHead with Calibrated Brier Loss,
runs true Split-Conformal calibration, and evaluates on held-out test split.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling_s1 import S1DecisionModel
from src.tokenizer_utils import (
    encode_decision_batch,
    format_decision_prompt,
    load_tokenizer_checked,
    save_tokenizer_checked,
)
from src.losses import CalibratedDecisionLoss
from src.conformal import ConformalDecisionCalibrator
from src.provenance import sha256_model_directory, sha256_tokenizer

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


def repo_relative_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def normalized_state(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def set_training_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_disjoint_training_splits(train_dataset, calibration_dataset, test_dataset):
    """Refuse to train a supposedly disjoint model when state groups overlap."""
    state_sets = {
        "train": {normalized_state(item["state"]) for item in train_dataset.samples},
        "calibration": {normalized_state(item["state"]) for item in calibration_dataset.samples},
        "test": {normalized_state(item["state"]) for item in test_dataset.samples},
    }
    intersections = {
        "train_calibration": state_sets["train"] & state_sets["calibration"],
        "train_test": state_sets["train"] & state_sets["test"],
        "calibration_test": state_sets["calibration"] & state_sets["test"],
    }
    failures = {name: len(values) for name, values in intersections.items() if values}
    if failures:
        detail = ", ".join(f"{name}={count}" for name, count in failures.items())
        raise ValueError(f"Normalized state overlap detected in training inputs: {detail}")


class S1DecisionDataset(Dataset):
    def __init__(self, data_path: Path):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                self.samples = json.loads(content)
            else:
                for line in content.splitlines():
                    if line.strip():
                        self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Dict], tokenizer, device, shuffle_options: bool = False, max_length: int = 2048):
    states = [item["state"] for item in batch]
    questions = [item["question"] for item in batch]

    options_list = []
    targets = []
    for item in batch:
        opts = list(item.get("candidates", item.get("options", [])))
        t_idx = item["target_idx"]

        if shuffle_options and len(opts) > 1:
            # Dynamically permute option order to completely break positional bias / collapse
            indices = list(range(len(opts)))
            random.shuffle(indices)
            shuffled_opts = [opts[i] for i in indices]
            new_target_idx = indices.index(t_idx)
            options_list.append(shuffled_opts)
            targets.append(new_target_idx)
        else:
            options_list.append(opts)
            targets.append(t_idx)
        
    encoded = encode_decision_batch(
        tokenizer=tokenizer,
        states=states,
        questions=questions,
        options_per_sample=options_list,
        targets=targets,
        max_length=max_length,
        device=device,
    )
    return encoded


def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                marker_indices=batch["marker_indices"],
                marker_mask=batch["marker_mask"],
                option_spans=batch.get("option_spans"),
            )
            loss, _ = loss_fn(
                logits=outputs["logits"],
                probs=outputs["probs"],
                escalate_risk=outputs["escalate_risk"],
                targets=batch["targets"],
                mask=batch["marker_mask"],
            )
            total_loss += loss.item() * len(batch["targets"])

            preds = outputs["best_choice_idx"]
            correct += (preds == batch["targets"]).sum().item()
            total += len(batch["targets"])
            
    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def train():
    repo_root = Path(__file__).resolve().parent.parent
    base_model_default = os.environ.get("BASE_MODEL_PATH", "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct")
    parser = argparse.ArgumentParser(description="Train S1 Decision Model")
    parser.add_argument("--base_model", type=str, default=base_model_default)
    parser.add_argument("--train_file", type=str, default=str(repo_root / "data/disjoint_v6/train_v6_disjoint.json"))
    parser.add_argument("--calib_file", type=str, default=str(repo_root / "data/disjoint_v6/calib_v6_disjoint.json"))
    parser.add_argument("--test_file", type=str, default=str(repo_root / "data/disjoint_v6/test_v6_disjoint.json"))
    parser.add_argument("--output_dir", type=str, default=str(repo_root / "output/s1_model_v6"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("epochs, batch_size, and grad_accum_steps must be positive integers")
    set_training_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"[*] Training on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    base_model_path = Path(args.base_model).expanduser().resolve()
    base_model_sha256 = sha256_model_directory(base_model_path)

    # 1. Load Tokenizer
    tokenizer = load_tokenizer_checked(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_sha256 = sha256_tokenizer(tokenizer)

    # 2. Build Backbone with LoRA & Gradient Checkpointing
    config = AutoConfig.from_pretrained(str(base_model_path))
    hidden_dim = getattr(config, "hidden_size", 896)

    backbone = AutoModel.from_pretrained(
        str(base_model_path),
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
    if device == "cuda":
        backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.print_trainable_parameters()


    # 3. Assemble Complete S1 Model
    model = S1DecisionModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_inter_layers=2,
    ).to(device)
    
    if device == "cuda":
        model.decision_head.to(dtype=torch.bfloat16)

    # 4. Prepare Datasets and Loaders
    train_data_path = Path(args.train_file).resolve()
    calibration_data_path = Path(args.calib_file).resolve()
    test_data_path = Path(args.test_file).resolve()
    train_file_metadata = repo_relative_path(train_data_path, repo_root)
    calibration_file_metadata = repo_relative_path(calibration_data_path, repo_root)
    test_file_metadata = repo_relative_path(test_data_path, repo_root)
    train_data_sha256 = sha256_file(train_data_path)
    calibration_data_sha256 = sha256_file(calibration_data_path)
    test_data_sha256 = sha256_file(test_data_path)

    train_dataset = S1DecisionDataset(train_data_path)
    calib_dataset = S1DecisionDataset(calibration_data_path)
    test_dataset = S1DecisionDataset(test_data_path)
    validate_disjoint_training_splits(train_dataset, calib_dataset, test_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, device, shuffle_options=True, max_length=2048),
    )
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

    # 5. Optimizer and Scheduler
    param_groups = [
        {"params": model.decision_head.parameters(), "lr": args.lr_head},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_lora},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    if not len(train_loader) or not len(calib_loader) or not len(test_loader):
        raise ValueError("train, calibration, and test splits must all be non-empty")
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    loss_fn = CalibratedDecisionLoss(lambda_brier=0.5, lambda_esc=0.3)

    # Initial Validation
    val_loss, val_acc = evaluate(model, calib_loader, loss_fn, device)
    logger.info(f"[*] Pre-training Calib Evaluation -> Loss: {val_loss:.4f} | Accuracy: {val_acc:.1%}")

    # Output directory setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "s1_decision_weights.pt"

    def save_checkpoint():
        filtered_lora = {k: v for k, v in model.backbone.state_dict().items() if 'lora' in k}
        save_tokenizer_checked(tokenizer, output_dir, expected_sha256=tokenizer_sha256)
        torch.save({
            "backbone_lora": filtered_lora,
            "decision_head": model.decision_head.state_dict(),
            "config": {
                "hidden_dim": hidden_dim,
                "num_heads": 8,
                "num_inter_layers": 2,
                "base_model": str(base_model_path),
                "base_model_sha256": base_model_sha256,
                "train_file": train_file_metadata,
                "train_data_sha256": train_data_sha256,
                "calib_file": calibration_file_metadata,
                "calibration_data_sha256": calibration_data_sha256,
                "test_file": test_file_metadata,
                "test_data_sha256": test_data_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "seed": args.seed,
            }
        }, save_path)

    # 6. Training Loop
    effective_bs = args.batch_size * args.grad_accum_steps
    logger.info(f"[*] Starting training for {args.epochs} epochs ({total_steps} optimizer steps, effective batch_size={effective_bs})...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                marker_indices=batch["marker_indices"],
                marker_mask=batch["marker_mask"],
                option_spans=batch.get("option_spans"),
            )
            
            loss, details = loss_fn(
                logits=outputs["logits"],
                probs=outputs["probs"],
                escalate_risk=outputs["escalate_risk"],
                targets=batch["targets"],
                mask=batch["marker_mask"],
            )
            
            loss_scaled = loss / args.grad_accum_steps
            loss_scaled.backward()
            
            epoch_loss += loss.item()
            step_count += 1
            
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (args.grad_accum_steps * 10) == 0:
                logger.info(
                    f"Epoch {epoch}/{args.epochs} [Step {step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} (CE: {details['ce_loss']:.4f}, Brier: {details['brier_loss']:.4f}, Margin: {details.get('margin_loss', 0.0):.4f})"
                )
                
        val_loss, val_acc = evaluate(model, calib_loader, loss_fn, device)
        logger.info(f"[Epoch {epoch} Summary] Train Loss: {epoch_loss/step_count:.4f} | Calib Loss: {val_loss:.4f} | Calib Acc: {val_acc:.1%}")
        # Save after every epoch
        save_checkpoint()
        logger.info(f"  -> Checkpoint updated: {save_path} ({save_path.stat().st_size / 1024 / 1024:.2f} MB)")

    elapsed = time.time() - start_time
    logger.info(f"[PASS] Training completed in {elapsed:.1f}s!")

    # 7. True Split-Conformal Calibration on Calib Set (15%)
    logger.info(f"\n[*] Running True Split-Conformal Calibration on Calib Split ({len(calib_dataset)} samples)...")
    model.eval()
    all_calib_probs = []
    all_calib_targets = []
    all_calib_masks = []
    max_options = max(
        len(item.get("candidates", item.get("options", [])))
        for item in list(calib_dataset.samples) + list(test_dataset.samples)
    )
    
    with torch.no_grad():
        for b in calib_loader:
            out = model(
                b["input_ids"],
                b["attention_mask"],
                marker_indices=b["marker_indices"],
                marker_mask=b["marker_mask"],
                option_spans=b.get("option_spans"),
            )
            probs = out["probs"].cpu()
            if probs.shape[1] < max_options:
                probs = torch.cat(
                    (probs, torch.zeros((probs.shape[0], max_options - probs.shape[1]), dtype=probs.dtype)),
                    dim=1,
                )
            all_calib_probs.append(probs)
            all_calib_targets.append(b["targets"].cpu())
            all_calib_masks.append(
                torch.cat(
                    (
                        b["marker_mask"].cpu(),
                        torch.zeros(
                            (b["marker_mask"].shape[0], max_options - b["marker_mask"].shape[1]),
                            dtype=torch.bool,
                        ),
                    ),
                    dim=1,
                )
                if b["marker_mask"].shape[1] < max_options
                else b["marker_mask"].cpu()
            )
            
    calib_probs_t = torch.cat(all_calib_probs, dim=0)
    calib_targets_t = torch.cat(all_calib_targets, dim=0)

    calibrator = ConformalDecisionCalibrator(alpha=0.05)
    q_hat = calibrator.fit(calib_probs_t, calib_targets_t)

    with open(save_path, "rb") as wf:
        ckpt_sha = hashlib.sha256(wf.read()).hexdigest()
    calib_file_path = output_dir / "conformal_calibration.json"
    calibration_metadata = {
        "calibrated_split": calibration_file_metadata,
        "calibration_data_sha256": calibration_data_sha256,
        "train_split": train_file_metadata,
        "train_data_sha256": train_data_sha256,
        "test_split": test_file_metadata,
        "test_data_sha256": test_data_sha256,
        "base_model": str(base_model_path),
        "base_model_sha256": base_model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "seed": args.seed,
    }
    calibrator.save(
        calib_file_path,
        checkpoint_sha256=ckpt_sha,
        metadata=calibration_metadata,
    )
    logger.info(f"[PASS] Calibration fitted: q_hat={q_hat:.4f} (bound to SHA-256: {ckpt_sha[:16]}...), saved to {calib_file_path}")

    # 8. Unbiased Evaluation on Held-Out Test Split (15%)
    logger.info(f"\n[*] Evaluating on State-Disjoint Test Split ({len(test_dataset)} samples)...")
    all_test_probs = []
    all_test_targets = []
    all_test_masks = []
    
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
            if probs.shape[1] < max_options:
                probs = torch.cat(
                    (probs, torch.zeros((probs.shape[0], max_options - probs.shape[1]), dtype=probs.dtype)),
                    dim=1,
                )
            all_test_probs.append(probs)
            all_test_targets.append(b["targets"].cpu())
            all_test_masks.append(
                torch.cat(
                    (
                        b["marker_mask"].cpu(),
                        torch.zeros(
                            (b["marker_mask"].shape[0], max_options - b["marker_mask"].shape[1]),
                            dtype=torch.bool,
                        ),
                    ),
                    dim=1,
                )
                if b["marker_mask"].shape[1] < max_options
                else b["marker_mask"].cpu()
            )

            
    test_probs_t = torch.cat(all_test_probs, dim=0)
    test_targets_t = torch.cat(all_test_targets, dim=0)
    test_masks_t = torch.cat(all_test_masks, dim=0)
    
    test_metrics = calibrator.evaluate_coverage(
        test_probs_t,
        test_targets_t,
        alpha=0.05,
        valid_mask=test_masks_t,
    )
    masked_test_probs = test_probs_t.masked_fill(~test_masks_t, -1.0)
    test_preds = torch.argmax(masked_test_probs, dim=-1)
    test_acc = (test_preds == test_targets_t).float().mean().item()
    
    logger.info("=" * 60)
    logger.info("    UNBIASED HELD-OUT TEST EVALUATION (ZERO TEMPLATE OVERLAP)")
    logger.info("=" * 60)
    logger.info(f"  Top-1 Accuracy:                 {test_acc:.2%}")
    logger.info(f"  Empirical Conformal Coverage:   {test_metrics['empirical_coverage']:.2%} (Target: >= 95.0%)")
    logger.info(f"  Act Coverage Rate (Auto-pass):  {test_metrics['act_coverage_rate']:.2%}")
    logger.info(f"  Selective Risk on Act:          {test_metrics['selective_risk_on_act']:.2%}")
    logger.info(f"  Abstention / Rejection Rate:    {test_metrics['abstention_rate']:.2%}")
    logger.info(f"  Average Prediction Set Size:    {test_metrics['average_set_size']:.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
