# -*- coding: utf-8 -*-
"""
Official V6 Production Checkpoint Downloader & Integrity Verifier for Millennium-Jev.
Ensures that all runtime deployments and academic benchmarks use the cryptographically
verified V6 model artifact (SHA-256: eaf07edd808cecf43473a13e6a331f57bc8d81696f2a1ed7d82dbc7467e01191).
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_v6")

V6_EXPECTED_SHA256 = "eaf07edd808cecf43473a13e6a331f57bc8d81696f2a1ed7d82dbc7467e01191"
HF_REPO_ID = "sunziqin/millennium-jev-0.5b"

REQUIRED_FILES = [
    "s1_decision_weights.pt",
    "conformal_calibration.json",
    "config.json",
    "model_metadata.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 of file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_local_checkpoint(target_dir: Path) -> bool:
    """Verify local files exist and match the official SHA-256 fingerprint."""
    weight_file = target_dir / "s1_decision_weights.pt"
    calib_file = target_dir / "conformal_calibration.json"

    if not weight_file.exists():
        logger.info(f"[-] Checkpoint file missing: {weight_file}")
        return False
    if not calib_file.exists():
        logger.info(f"[-] Calibration file missing: {calib_file}")
        return False

    actual_hash = sha256_file(weight_file)
    if actual_hash != V6_EXPECTED_SHA256:
        logger.warning(
            f"[!] Hash mismatch for {weight_file.name}!\n"
            f"    Expected: {V6_EXPECTED_SHA256}\n"
            f"    Found:    {actual_hash}\n"
            f"    This may be an uncalibrated, legacy (v4/v5) or corrupt checkpoint."
        )
        return False

    try:
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        if calib_data.get("checkpoint_sha256") != V6_EXPECTED_SHA256:
            logger.warning("[!] Calibration file is not cryptographically bound to V6 weights.")
            return False
    except Exception as e:
        logger.warning(f"[!] Failed to parse calibration file: {e}")
        return False

    logger.info(f"[+] Local V6 checkpoint verified successfully! (SHA-256: {actual_hash[:16]}...)")
    return True


def download_from_hf(target_dir: Path, repo_id: str = HF_REPO_ID, hf_token: Optional[str] = None):
    """Download official weights from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub library is not installed.\n"
            "Please install it via: pip install huggingface_hub\n"
            "Or download manually from: https://huggingface.co/" + repo_id
        )
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading official Millennium-Jev 0.5B (V6) from Hugging Face: {repo_id} ...")

    token = hf_token or os.environ.get("HF_TOKEN")

    for filename in REQUIRED_FILES:
        try:
            logger.info(f"Downloading {filename} ...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                token=token,
            )
            logger.info(f"  Downloaded: {downloaded_path}")
        except Exception as e:
            logger.warning(f"  Failed to download {filename} from Hub: {e}")

    return verify_local_checkpoint(target_dir)


def main():
    parser = argparse.ArgumentParser(description="Download & Verify Official Millennium-Jev 0.5B Checkpoint")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="output/s1_model_v6",
        help="Local directory to store V6 checkpoint (default: output/s1_model_v6)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=HF_REPO_ID,
        help=f"Hugging Face repository ID (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local files pass cryptographic check",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()

    if not args.force and verify_local_checkpoint(target_dir):
        print("\n" + "=" * 80)
        print(" [SUCCESS] Millennium-Jev 0.5B (V6) is already installed and verified!")
        print(f" Directory:        {target_dir}")
        print(f" Checkpoint SHA:   {V6_EXPECTED_SHA256}")
        print(" Ready to run:     python scripts/run_live_broad_audit.py")
        print("=" * 80)
        return 0

    success = download_from_hf(target_dir, repo_id=args.repo_id)
    if success:
        print("\n" + "=" * 80)
        print(" [SUCCESS] Download completed and verified successfully!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print(" [NOTICE] Automated Hugging Face download was not completed.")
        print(f" If the Hugging Face repository {args.repo_id} is pending public push,")
        print(f" please place the official V6 weights into: {target_dir}")
        print(f" Target Checkpoint SHA-256: {V6_EXPECTED_SHA256}")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
