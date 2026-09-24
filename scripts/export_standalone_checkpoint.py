# -*- coding: utf-8 -*-
"""
Export Aegis-S1 Standalone Merged Checkpoint.
Fuses the LoRA adapter weights directly into the base Qwen2.5-0.5B backbone weights,
producing a standalone model checkpoint that requires no PEFT overhead at inference.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_standalone")


def export_standalone(
    model_dir: Path,
    base_model_path: Path,
    output_dir: Path,
    device: str = "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_file = model_dir / "s1_decision_weights.pt"
    calib_file = model_dir / "conformal_calibration.json"

    if not weight_file.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    logger.info(f"Loading checkpoint from {weight_file} ...")
    checkpoint = torch.load(weight_file, map_location="cpu")
    lora_state = checkpoint["backbone_lora"]
    head_state = checkpoint["decision_head"]

    logger.info(f"Loading base model from {base_model_path} ...")
    config = AutoConfig.from_pretrained(base_model_path)
    base_model = AutoModel.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float32,
    )

    sample_lora_a = next((v for k, v in lora_state.items() if "lora_A" in k), None)
    detected_r = sample_lora_a.shape[0] if sample_lora_a is not None else 32
    target_modules = sorted({k.split(".")[-2] for k in lora_state if "lora_A" in k or "lora_B" in k})

    lora_config = LoraConfig(
        r=detected_r,
        lora_alpha=detected_r * 2,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.load_state_dict(lora_state, strict=False)

    logger.info("Merging LoRA weights into base backbone ...")
    fused_backbone = peft_model.merge_and_unload()

    logger.info("Saving standalone fused weights and tokenizer ...")
    standalone_weight_file = output_dir / "s1_fused_weights.pt"
    torch.save(
        {
            "backbone": fused_backbone.state_dict(),
            "decision_head": head_state,
            "config": checkpoint.get("config", {}),
            "is_fused": True,
        },
        standalone_weight_file,
    )
    logger.info(f"Saved fused weights to {standalone_weight_file} ({standalone_weight_file.stat().st_size / 1e6:.1f} MB)")

    # Copy calibration artifact and tokenizer files
    shutil.copy2(calib_file, output_dir / "conformal_calibration.json")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Export completed successfully to {output_dir}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Standalone Aegis-S1 Model")
    parser.add_argument("--model_dir", type=str, default="output/s1_model_v6")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=os.environ.get("BASE_MODEL_PATH", "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--output_dir", type=str, default="output/s1_model_v6_standalone")
    args = parser.parse_args()

    export_standalone(
        model_dir=Path(args.model_dir),
        base_model_path=Path(args.base_model_path),
        output_dir=Path(args.output_dir),
    )
