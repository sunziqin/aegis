"""Shared provenance hashing and checkpoint/calibration consistency checks."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


SHA256_HEX_LENGTH = 64
PROVENANCE_FIELDS = (
    "base_model",
    "base_model_sha256",
    "tokenizer_sha256",
    "calibrated_split",
    "calibration_data_sha256",
    "train_split",
    "train_data_sha256",
    "test_split",
    "test_data_sha256",
)
PROVENANCE_HASH_FIELDS = {
    "base_model_sha256",
    "tokenizer_sha256",
    "calibration_data_sha256",
    "train_data_sha256",
    "test_data_sha256",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_model_directory(model_dir: Path) -> str:
    """Hash every file in a base-model directory using a stable path manifest."""
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {root}")
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"Base model directory is empty: {root}")

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_tokenizer(tokenizer) -> str:
    """Hash vocabulary and behavior-affecting tokenizer settings."""
    try:
        tokenizer_json = tokenizer.to_json()
    except (AttributeError, TypeError, ValueError):
        tokenizer_json = None

    behavior = {
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "clean_up_tokenization_spaces": getattr(tokenizer, "clean_up_tokenization_spaces", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "added_vocab": getattr(tokenizer, "get_added_vocab", lambda: {})(),
    }
    if tokenizer_json is None:
        behavior["vocab"] = getattr(tokenizer, "get_vocab")()
    else:
        behavior["tokenizer_json"] = tokenizer_json
    serialized = json.dumps(behavior, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def provenance_key(value: Any) -> str:
    """Normalize a path-like provenance value for cross-artifact comparison."""
    return str(Path(str(value)).expanduser().resolve()).casefold()


def validate_provenance_pair(checkpoint: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Require matching complete provenance in checkpoint config and calibration metadata."""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise RuntimeError("Checkpoint has no config provenance; retrain on the current disjoint_v6 splits.")
    if not isinstance(metadata, dict):
        raise RuntimeError("Calibration artifact must contain provenance metadata")

    config = checkpoint["config"]
    missing_metadata = [field for field in PROVENANCE_FIELDS if not metadata.get(field)]
    if missing_metadata:
        raise RuntimeError(
            "Calibration artifact is missing provenance fields: "
            + ", ".join(missing_metadata)
            + ". Retrain and recalibrate on the current disjoint_v6 splits."
        )
    config_fields = {
        "base_model": "base_model",
        "base_model_sha256": "base_model_sha256",
        "tokenizer_sha256": "tokenizer_sha256",
        "train_split": "train_file",
        "train_data_sha256": "train_data_sha256",
        "calibrated_split": "calib_file",
        "calibration_data_sha256": "calibration_data_sha256",
        "test_split": "test_file",
        "test_data_sha256": "test_data_sha256",
    }
    missing_config = [config_field for config_field in config_fields.values() if not config.get(config_field)]
    if missing_config:
        raise RuntimeError(
            "Checkpoint config is missing provenance fields: "
            + ", ".join(missing_config)
            + ". Retrain on the current disjoint_v6 splits."
        )

    for field in PROVENANCE_HASH_FIELDS:
        for source, value in (("calibration", metadata[field]), ("checkpoint", config[config_fields[field]])):
            text = str(value)
            if len(text) != SHA256_HEX_LENGTH or any(char not in "0123456789abcdefABCDEF" for char in text):
                raise RuntimeError(f"{source} provenance field {field} must be a 64-character SHA-256 hash")

    for metadata_field, config_field in config_fields.items():
        metadata_value = metadata[metadata_field]
        config_value = config[config_field]
        if metadata_field in PROVENANCE_HASH_FIELDS:
            matches = str(metadata_value).casefold() == str(config_value).casefold()
        else:
            # Paths are descriptive metadata. They can legitimately differ when a
            # checkpoint is calibrated or deployed on another host; the paired
            # content hashes are the binding identity checks.
            matches = bool(metadata_value) and bool(config_value)
        if not matches:
            raise RuntimeError(f"Checkpoint/calibration provenance mismatch for {metadata_field}")


def resolve_provenance_file(value: Any, root_dir: Path) -> Path:
    """Resolve an absolute or repository-relative data path and require it to exist."""
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(root_dir) / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise RuntimeError(f"Provenance data file not found: {candidate}")
    return candidate


__all__ = [
    "PROVENANCE_FIELDS",
    "PROVENANCE_HASH_FIELDS",
    "provenance_key",
    "resolve_provenance_file",
    "sha256_file",
    "sha256_model_directory",
    "sha256_tokenizer",
    "validate_provenance_pair",
]
