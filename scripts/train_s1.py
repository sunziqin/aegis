# -*- coding: utf-8 -*-
"""
Training Pipeline for Project Aegis-S1 (Non-Autoregressive Decision Model).
Trains LoRA on Qwen2.5-0.5B backbone + DynamicOptionMarkerHead with Calibrated Brier Loss.
"""

import argparse
import json
import logging
import math
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
from src.tokenizer_utils import format_decision_prompt, encode_decision_batch
from src.losses import CalibratedDecisionLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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


def collate_fn(batch: List[Dict], tokenizer, device):
    prompts = []
    options_list = []
    targets = []
    
    for item in batch:
        opts = item.get("candidates", item.get("options", []))
        p = format_decision_prompt(item["state"], item["question"], opts)
        prompts.append(p)
        options_list.append(opts)
        targets.append(item["target_idx"])
        
    encoded = encode_decision_batch(
        tokenizer=tokenizer,
        batch_prompts=prompts,
        options_per_sample=options_list,
        targets=targets,
        max_length=512,
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
    parser = argparse.ArgumentParser(description="Train S1 Decision Model")
    parser.add_argument("--base_model", type=str, default="E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="E:/s1-decision-model/data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="E:/s1-decision-model/data/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="E:/s1-decision-model/output/s1_model_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[*] Training on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build Backbone with LoRA
    config = AutoConfig.from_pretrained(args.base_model)
    hidden_dim = getattr(config, "hidden_size", 896)
    
    backbone = AutoModel.from_pretrained(
        args.base_model,
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
    train_dataset = S1DecisionDataset(Path(args.train_file))
    val_dataset = S1DecisionDataset(Path(args.val_file))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
    )

    # 5. Optimizer and Scheduler
    param_groups = [
        {"params": model.decision_head.parameters(), "lr": args.lr_head},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_lora},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    loss_fn = CalibratedDecisionLoss(lambda_brier=0.5, lambda_esc=0.3)

    # Initial Validation
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    logger.info(f"[*] Pre-training Validation -> Loss: {val_loss:.4f} | Accuracy: {val_acc:.1%}")

    # 6. Training Loop
    logger.info(f"[*] Starting training for {args.epochs} epochs ({total_steps} steps)...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                marker_indices=batch["marker_indices"],
                marker_mask=batch["marker_mask"],
            )
            
            loss, details = loss_fn(
                logits=outputs["logits"],
                probs=outputs["probs"],
                escalate_risk=outputs["escalate_risk"],
                targets=batch["targets"],
                mask=batch["marker_mask"],
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step_count += 1
            
            if (step + 1) % 25 == 0:
                logger.info(
                    f"Epoch {epoch}/{args.epochs} [Step {step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} (CE: {details['ce_loss']:.4f}, Brier: {details['brier_loss']:.4f})"
                )
                
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        logger.info(f"[Epoch {epoch} Summary] Train Loss: {epoch_loss/step_count:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")

    elapsed = time.time() - start_time
    logger.info(f"[PASS] Training completed in {elapsed:.1f}s!")

    # 7. Save Model Weights & Tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save standalone weights
    save_path = output_dir / "s1_decision_weights.pt"
    torch.save({
        "backbone_lora": model.backbone.state_dict(),
        "decision_head": model.decision_head.state_dict(),
        "config": {
            "hidden_dim": hidden_dim,
            "num_heads": 8,
            "num_inter_layers": 2,
            "base_model": args.base_model,
        }
    }, save_path)
    
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[PASS] Model checkpoint and tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    train()
