from pathlib import Path

import pytest

from src.provenance import (
    provenance_key,
    sha256_file,
    sha256_model_directory,
    sha256_tokenizer,
    validate_provenance_pair,
)
from src.conformal import ConformalDecisionCalibrator
from open_s1 import _validate_artifact_provenance


class _Tokenizer:
    padding_side = "right"
    truncation_side = "right"
    model_max_length = 2048
    clean_up_tokenization_spaces = False
    special_tokens_map = {"eos_token": "<eos>"}

    def get_vocab(self):
        return {"a": 0, "b": 1}

    def get_added_vocab(self):
        return {"<marker>": 2}


def _complete_pair(tmp_path: Path):
    digest = "a" * 64
    base = tmp_path / "base-model"
    train = tmp_path / "train.json"
    calib = tmp_path / "calib.json"
    test = tmp_path / "test.json"
    for path in (train, calib, test):
        path.write_text("[]", encoding="utf-8")
    metadata = {
        "base_model": str(base),
        "base_model_sha256": digest,
        "tokenizer_sha256": digest,
        "calibrated_split": str(calib),
        "calibration_data_sha256": digest,
        "train_split": str(train),
        "train_data_sha256": digest,
        "test_split": str(test),
        "test_data_sha256": digest,
    }
    config = {
        "base_model": str(base),
        "base_model_sha256": digest,
        "tokenizer_sha256": digest,
        "calib_file": str(calib),
        "calibration_data_sha256": digest,
        "train_file": str(train),
        "train_data_sha256": digest,
        "test_file": str(test),
        "test_data_sha256": digest,
    }
    return {"config": config}, metadata


def test_model_directory_hash_changes_when_any_file_changes(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.bin").write_bytes(b"weights-v1")
    first = sha256_model_directory(model_dir)
    (model_dir / "weights.bin").write_bytes(b"weights-v2")
    assert sha256_model_directory(model_dir) != first


def test_tokenizer_hash_includes_behavior_settings():
    tokenizer = _Tokenizer()
    first = sha256_tokenizer(tokenizer)
    tokenizer.padding_side = "left"
    assert sha256_tokenizer(tokenizer) != first


def test_provenance_pair_requires_matching_hashes_and_paths(tmp_path):
    checkpoint, metadata = _complete_pair(tmp_path)
    validate_provenance_pair(checkpoint, metadata)

    checkpoint["config"]["base_model_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate_provenance_pair(checkpoint, metadata)

    assert provenance_key(tmp_path / "base-model") == provenance_key(str(tmp_path / "base-model"))


def test_sdk_provenance_rehashes_referenced_data(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "weights.bin").write_bytes(b"base")
    data_paths = {name: tmp_path / f"{name}.json" for name in ("train", "calib", "test")}
    for path in data_paths.values():
        path.write_text("[]", encoding="utf-8")
    base_hash = sha256_model_directory(base)
    data_hashes = {name: sha256_file(path) for name, path in data_paths.items()}
    digest = "c" * 64
    metadata = {
        "base_model": str(base),
        "base_model_sha256": base_hash,
        "tokenizer_sha256": digest,
        "calibrated_split": str(data_paths["calib"]),
        "calibration_data_sha256": data_hashes["calib"],
        "train_split": str(data_paths["train"]),
        "train_data_sha256": data_hashes["train"],
        "test_split": str(data_paths["test"]),
        "test_data_sha256": data_hashes["test"],
    }
    checkpoint = {
        "config": {
            "base_model": str(base),
            "base_model_sha256": base_hash,
            "tokenizer_sha256": digest,
            "calib_file": str(data_paths["calib"]),
            "calibration_data_sha256": data_hashes["calib"],
            "train_file": str(data_paths["train"]),
            "train_data_sha256": data_hashes["train"],
            "test_file": str(data_paths["test"]),
            "test_data_sha256": data_hashes["test"],
        }
    }
    calibrator = ConformalDecisionCalibrator()
    calibrator.metadata = metadata
    _validate_artifact_provenance(
        calibrator,
        checkpoint,
        tokenizer_sha256=digest,
        base_model_sha256=base_hash,
        base_model_path=base,
        root_dir=tmp_path,
    )

    data_paths["train"].write_text("[tampered]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="data hash mismatch"):
        _validate_artifact_provenance(
            calibrator,
            checkpoint,
            tokenizer_sha256=digest,
            base_model_sha256=base_hash,
            base_model_path=base,
            root_dir=tmp_path,
        )
