# -*- coding: utf-8 -*-
"""
Official 1.5B Flagship Checkpoint Downloader & Integrity Verifier for Millennium-Jev.
Ensures that all runtime deployments and academic benchmarks use the cryptographically
verified 1.5B flagship model artifact (SHA-256: 458c3b73157e05a99ae53b0a2e1eb090b3d434bc5facfb482b1fd884968f177d).
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
logger = logging.getLogger("download_1.5b")

V15B_EXPECTED_SHA256 = "458c3b73157e05a99ae53b0a2e1eb090b3d434bc5facfb482b1fd884968f177d"
HF_REPO_ID = "sunziqin/millennium-jev-1.5b"

REQUIRED_FILES = [
    "s1_decision_weights.pt",
    "conformal_calibration.json",
    "config.json",
    "model_metadata.json",
    "strictly_disjoint_evaluation_report.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
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
    if actual_hash != V15B_EXPECTED_SHA256:
        logger.warning(
            f"[!] Hash mismatch for {weight_file.name}!\n"
            f"    Expected: {V15B_EXPECTED_SHA256}\n"
            f"    Found:    {actual_hash}\n"
            f"    This checkpoint is either corrupt or modified."
        )
        return False

    logger.info(f"[+] Verified official 1.5B Checkpoint SHA-256: {actual_hash[:16]}... (MATCH)")
    return True


def download_from_hf(target_dir: Path, token: Optional[str] = None) -> bool:
    """Download official weights from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub package is required for downloading.\n"
            "Install it via: pip install huggingface_hub"
        )
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to Hugging Face Hub repo: {HF_REPO_ID} ...")

    for filename in REQUIRED_FILES:
        logger.info(f"Downloading {filename} ...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(target_dir),
                token=token or os.environ.get("HF_TOKEN"),
            )
            logger.info(f"  -> {filename} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False

    logger.info("All files downloaded. Running cryptographic integrity verification ...")
    return verify_local_checkpoint(target_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify official Millennium-Jev 1.5B Flagship weights."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="output/s1_model_1.5b",
        help="Local target directory to save the checkpoint files.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (optional if repo is public).",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing local files without downloading.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    target_dir = repo_root / args.target_dir

    print("=" * 65)
    print("   MILLENNIUM-JEV 1.5B FLAGSHIP CHECKPOINT VERIFIER")
    print("=" * 65)
    print(f"Target Directory: {target_dir}")
    print(f"Expected SHA-256: {V15B_EXPECTED_SHA256}")
    print(f"Hugging Face:     https://huggingface.co/{HF_REPO_ID}")
    print("=" * 65)

    if args.verify_only:
        if verify_local_checkpoint(target_dir):
            print("\n[SUCCESS] Local 1.5B checkpoint is genuine and verified!")
            sys.exit(0)
        else:
            print("\n[FAIL] Local 1.5B checkpoint verification failed!")
            sys.exit(1)

    if verify_local_checkpoint(target_dir):
        print("\n[+] Official 1.5B checkpoint is already present and fully verified!")
        sys.exit(0)

    print("\n[*] Local checkpoint missing or invalid. Downloading from Hugging Face...")
    if download_from_hf(target_dir, token=args.token):
        print("\n[SUCCESS] Millennium-Jev 1.5B checkpoint downloaded and verified successfully!")
        sys.exit(0)
    else:
        print("\n[FAIL] Failed to download or verify Millennium-Jev 1.5B checkpoint.")
        sys.exit(1)


if __name__ == "__main__":
    main()
