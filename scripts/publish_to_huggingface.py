# -*- coding: utf-8 -*-
"""
Upload Millennium-Jev 0.5B (V6) Official Weights to Hugging Face Hub.
Usage:
  python scripts/publish_to_huggingface.py --repo_id sunziqin/millennium-jev-0.5b
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_hf")

repo_root = Path(__file__).resolve().parent.parent


def publish_model(model_dir: Path, repo_id: str, hf_token: str = None):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub is not installed. Please run: pip install huggingface_hub")
        return False

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error(
            "Hugging Face token not found. Please provide --token or set the HF_TOKEN environment variable."
        )
        return False

    api = HfApi(token=token)

    logger.info(f"Ensuring repository exists: {repo_id} ...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
        logger.info(f"[+] Repository {repo_id} confirmed!")
    except Exception as e:
        logger.warning(f"Note on create_repo: {e}")

    logger.info(f"Uploading files from {model_dir} to {repo_id} ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release Millennium-Jev 0.5B (V6) official weights with conformal calibration",
    )

    logger.info(f"[+] Successfully uploaded Millennium-Jev 0.5B (V6) to https://huggingface.co/{repo_id}!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Millennium-Jev Checkpoint to Hugging Face")
    parser.add_argument("--model_dir", type=str, default="output/s1_model_v6")
    parser.add_argument("--repo_id", type=str, default="sunziqin/millennium-jev-0.5b")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    model_dir = repo_root / args.model_dir
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    success = publish_model(model_dir=model_dir, repo_id=args.repo_id, hf_token=args.token)
    sys.exit(0 if success else 1)
