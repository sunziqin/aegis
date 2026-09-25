# -*- coding: utf-8 -*-
"""
Upload Millennium-Jev 1.5B checkpoint to Hugging Face Hub using standard LFS protocol.
"""
import os
import sys
import time
from pathlib import Path

# Force standard LFS protocol instead of xet
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "sunziqin/millennium-jev-1.5b"
FILE_PATH = r"E:\s1-decision-model\output\s1_model_1.5b\s1_decision_weights.pt"

def main():
    if not os.path.exists(FILE_PATH):
        print(f"[!] Error: File {FILE_PATH} does not exist!")
        sys.exit(1)

    file_size_mb = os.path.getsize(FILE_PATH) / (1024 * 1024)
    print(f"[*] Target file: {FILE_PATH} ({file_size_mb:.2f} MB)")
    print(f"[*] Target repo: https://huggingface.co/{REPO_ID}")
    print("[*] Using standard Hugging Face LFS with retries...")

    api = HfApi()

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[*] Upload attempt {attempt}/{max_retries} starting...")
            start_t = time.time()
            res = api.upload_file(
                path_or_fileobj=FILE_PATH,
                path_in_repo="s1_decision_weights.pt",
                repo_id=REPO_ID,
                token=TOKEN,
                commit_message="Add official Millennium-Jev 1.5B (Flagship) weights",
            )
            elapsed = time.time() - start_t
            print(f"[+] SUCCESS! Upload completed in {elapsed:.1f}s.")
            print(f"[+] Commit info: {res}")
            return
        except Exception as e:
            print(f"[-] Attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                wait_s = 5 * attempt
                print(f"[*] Sleeping {wait_s}s before next retry...")
                time.sleep(wait_s)
            else:
                print("[!] All attempts exhausted.")
                sys.exit(1)

if __name__ == "__main__":
    main()
