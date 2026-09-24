# -*- coding: utf-8 -*-
"""
NAS Manager for Aegis-S1 Training on RTX 3060 12GB (fnOS).
Handles SSH execution, SFTP file synchronization, and Docker container lifecycle.
"""

import os
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
import paramiko


_HASH_CHUNK_SIZE = 1024 * 1024
_PASSWORD_PROMPT_RE = re.compile(r"^\s*(?:\[sudo\]\s*)?password for [^:\r\n]+:\s*$", re.IGNORECASE)

NAS_HOST = os.environ.get("S1_NAS_HOST", "127.0.0.1")
NAS_PORT = int(os.environ.get("S1_NAS_PORT", "22"))
REMOTE_WORKSPACE = os.environ.get("S1_NAS_REMOTE_WORKSPACE", "/workspace/aegis")
LOCAL_ROOT = Path(os.environ.get("S1_REPO_ROOT", Path(__file__).resolve().parent.parent))


def _credentials() -> Tuple[str, str]:
    user = os.environ.get("S1_NAS_USER")
    password = os.environ.get("S1_NAS_PASSWORD")
    missing = [
        name for name, value in (("S1_NAS_USER", user), ("S1_NAS_PASSWORD", password)) if not value
    ]
    if missing:
        raise RuntimeError("Set the NAS credentials in environment variables: " + ", ".join(missing))
    return user, password


def get_client() -> paramiko.SSHClient:
    user, password = _credentials()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known_hosts = os.environ.get("S1_NAS_KNOWN_HOSTS")
    if known_hosts:
        client.load_host_keys(os.path.expanduser(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(NAS_HOST, port=NAS_PORT, username=user, password=password, timeout=15)
    return client


def _filter_expected_stderr(err: str, *, suppress_password_prompt: bool = False) -> str:
    """Remove only known shell noise while preserving actionable stderr."""
    kept = []
    for line in err.splitlines():
        if "Could not chdir to home" in line:
            continue
        if suppress_password_prompt and _PASSWORD_PROMPT_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def run_remote(cmd: str, sudo: bool = False, print_output: bool = True) -> Tuple[int, str, str]:
    client = get_client()
    password = _credentials()[1] if sudo else None
    full_cmd = f"sudo -S -p '' {cmd}" if sudo else cmd
    stdin, stdout, stderr = client.exec_command(full_cmd, get_pty=bool(sudo))
    if password is not None:
        stdin.write(password + "\n")
        stdin.flush()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    client.close()
    if print_output:
        if out.strip():
            safe_out = out.strip().encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
            print(safe_out)
        if err.strip():
            filtered_err = _filter_expected_stderr(err, suppress_password_prompt=sudo)
            if filtered_err.strip():
                safe_err = filtered_err.strip().encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
                print(f"[STDERR] {safe_err}")
    return exit_code, out, err


def sftp_upload_file(local_path: Path, remote_path: str):
    client = get_client()
    sftp = client.open_sftp()
    
    # Ensure remote directory exists
    remote_dir = os.path.dirname(remote_path).replace("\\", "/")
    run_remote(f"mkdir -p '{remote_dir}'", sudo=False, print_output=False)
    
    file_size_mb = local_path.stat().st_size / 1024 / 1024
    t0 = time.time()
    print(f"Uploading {local_path.name} ({file_size_mb:.2f} MB) -> {remote_path} ...", end="", flush=True)
    sftp.put(str(local_path), remote_path)
    sftp.close()
    client.close()
    elapsed = time.time() - t0
    speed = file_size_mb / max(0.01, elapsed)
    print(f" DONE in {elapsed:.1f}s ({speed:.1f} MB/s)")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_remote_file(sftp, remote_path: str) -> str:
    digest = hashlib.sha256()
    with sftp.open(remote_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sftp_sync_file_smart(sftp, local_path: Path, remote_path: str):
    """Upload unless the remote file has the same size and SHA-256 digest."""
    local_path = Path(local_path)
    file_size = local_path.stat().st_size
    file_size_mb = file_size / 1024 / 1024
    
    try:
        rem_stat = sftp.stat(remote_path)
        if rem_stat.st_size == file_size:
            local_sha256 = _sha256_file(local_path)
            try:
                remote_sha256 = _sha256_remote_file(sftp, remote_path)
            except (OSError, IOError, paramiko.SSHException):
                remote_sha256 = None
            if remote_sha256 == local_sha256:
                print(f"[SKIP] {local_path.name} already up-to-date ({file_size_mb:.2f} MB)")
                return
    except IOError:
        pass
        
    t0 = time.time()
    print(f"[UPLOAD] {local_path.name} ({file_size_mb:.2f} MB) -> {remote_path} ... ", end="", flush=True)
    sftp.put(str(local_path), remote_path)
    elapsed = time.time() - t0
    speed = file_size_mb / max(0.01, elapsed)
    print(f"DONE in {elapsed:.1f}s ({speed:.1f} MB/s)")


def sync_all_assets():
    """Syncs code, dataset v6, and Qwen2.5-0.5B base model to NAS."""
    client = get_client()
    sftp = client.open_sftp()
    
    # 1. Sync Code
    print("\n--- 1. SYNCING CODE ---")
    run_remote(f"mkdir -p {REMOTE_WORKSPACE}/code/src {REMOTE_WORKSPACE}/code/open_s1 {REMOTE_WORKSPACE}/code/scripts", sudo=False, print_output=False)
    
    local_root = LOCAL_ROOT
    for subdir in ["src", "open_s1", "scripts"]:
        sdir = local_root / subdir
        for f in sdir.glob("*.py"):
            rem_p = f"{REMOTE_WORKSPACE}/code/{subdir}/{f.name}"
            sftp_sync_file_smart(sftp, f, rem_p)
            
    # 2. Sync Datasets V6
    print("\n--- 2. SYNCING DATASETS V6 ---")
    run_remote(f"mkdir -p {REMOTE_WORKSPACE}/code/data/disjoint_v6", sudo=False, print_output=False)
    data_dir = local_root / "data" / "disjoint_v6"
    for name in ["train_v6_disjoint.json", "calib_v6_disjoint.json", "test_v6_disjoint.json"]:
        f = data_dir / name
        if f.exists():
            sftp_sync_file_smart(sftp, f, f"{REMOTE_WORKSPACE}/code/data/disjoint_v6/{name}")
        else:
            print(f"[WARN] Local dataset {f} does not exist!")

    # 3. Sync Base Model
    print("\n--- 3. SYNCING BASE MODEL (Qwen2.5-0.5B-Instruct) ---")
    base_model_dir = Path(
        os.environ.get("BASE_MODEL_PATH", "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct")
    ).expanduser()
    remote_base = f"{REMOTE_WORKSPACE}/models/base/Qwen2.5-0.5B-Instruct"
    run_remote(f"mkdir -p {remote_base}", sudo=False, print_output=False)
    
    for f in base_model_dir.iterdir():
        if f.is_file():
            sftp_sync_file_smart(sftp, f, f"{remote_base}/{f.name}")
            
    sftp.close()
    client.close()
    print("\n[ALL ASSETS SYNCED SUCCESSFULLY]")


def check_cuda():
    cmd = 'docker exec aegis-trainer /workspace/venv/bin/python -c "import torch; print(\'CUDA:\', torch.cuda.is_available(), \'|\', torch.cuda.get_device_name(0), \'|\', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), \'GB\'); print(\'PyTorch:\', torch.__version__)"'
    run_remote(cmd, sudo=True)


def install_deps():
    packages = ["transformers==4.50.0", "peft", "accelerate", "datasets", "scipy", "scikit-learn", "tqdm"]
    pkg_str = " ".join(packages)
    cmd = f"docker exec aegis-trainer /workspace/venv/bin/pip install --proxy http://127.0.0.1:7890 {pkg_str}"
    print(f"[*] Installing dependencies: {pkg_str} ...")
    run_remote(cmd, sudo=True)


def start_training():
    train_cmd = (
        "nohup /workspace/venv/bin/python -u /workspace/code/scripts/train_s1.py "
        "--base_model /workspace/models/base/Qwen2.5-0.5B-Instruct "
        "--train_file /workspace/code/data/disjoint_v6/train_v6_disjoint.json "
        "--calib_file /workspace/code/data/disjoint_v6/calib_v6_disjoint.json "
        "--test_file /workspace/code/data/disjoint_v6/test_v6_disjoint.json "
        "--output_dir /workspace/code/output/s1_model_v6 "
        "--epochs 3 "
        "--batch_size 4 "
        "--grad_accum_steps 4 "
        "> /workspace/code/output/training_v6.log 2>&1 &"
    )
    cmd = f'docker exec -d aegis-trainer sh -c "mkdir -p /workspace/code/output && {train_cmd}"'
    print("[*] Launching training in container background...")
    run_remote(cmd, sudo=True)
    time.sleep(2)
    print("\n[*] Verifying running process:")
    run_remote("docker exec aegis-trainer ps aux | grep train_s1", sudo=True)


def tail_log(lines: int = 50):
    cmd = f"docker exec aegis-trainer tail -n {lines} /workspace/code/output/training_v6.log"
    run_remote(cmd, sudo=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "status":
            print("=== NVIDIA GPU STATUS ===")
            run_remote("nvidia-smi")
            print("\n=== DOCKER STATUS ===")
            run_remote("docker ps", sudo=True)
            print("\n=== STORAGE VOL2 ===")
            run_remote("df -h /vol2")
        elif action == "sync":
            sync_all_assets()
        elif action == "check-cuda":
            check_cuda()
        elif action == "install-deps":
            install_deps()
        elif action == "start-train":
            start_training()
        elif action == "log":
            lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            tail_log(lines)
        elif action == "cmd":
            cmd = " ".join(sys.argv[2:])
            run_remote(cmd, sudo=False)
        elif action == "sudo":
            cmd = " ".join(sys.argv[2:])
            run_remote(cmd, sudo=True)
    else:
        print("Usage: python nas_manager.py [status | sync | check-cuda | install-deps | start-train | log [n] | cmd | sudo]")
