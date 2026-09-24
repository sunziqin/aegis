import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_v6_comprehensive import (
    load_calibration_artifact,
    load_lora_state_checked,
    sha256_file,
    validate_checkpoint_training,
    validate_disjoint_splits,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sample(state: str, target_idx: int = 0):
    return {
        "state": state,
        "question": "choose",
        "candidates": ["a", "b"],
        "target_idx": target_idx,
    }


def test_validate_disjoint_splits_uses_normalized_state():
    splits = {
        "train": [_sample("Train state")],
        "calibration": [_sample("calibration state")],
        "test": [_sample("  TEST   state ")],
    }
    provenance = validate_disjoint_splits(splits)
    assert provenance["verified_zero_normalized_state_overlap"] is True
    assert provenance["normalized_state_intersections"] == {
        "train_calibration": 0,
        "train_test": 0,
        "calibration_test": 0,
    }

    splits["test"] = [_sample(" train   state ")]
    with pytest.raises(ValueError, match="Normalized state overlap"):
        validate_disjoint_splits(splits)


def test_checkpoint_training_provenance_requires_matching_hash(tmp_path):
    train_file = tmp_path / "train.json"
    _write_json(train_file, [_sample("train")])
    train_hash = sha256_file(train_file)
    checkpoint = {
        "config": {
            "train_file": str(train_file),
            "train_data_sha256": train_hash,
        }
    }
    assert validate_checkpoint_training(checkpoint, train_file) == train_hash

    checkpoint["config"]["train_data_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="training data hash mismatch"):
        validate_checkpoint_training(checkpoint, train_file)


def test_lora_checkpoint_shape_and_key_validation():
    module = torch.nn.Module()
    module.lora_A = torch.nn.Linear(3, 2, bias=False)
    module.lora_B = torch.nn.Linear(2, 3, bias=False)
    state = {key: value.detach().clone() for key, value in module.state_dict().items()}
    load_lora_state_checked(module, state)

    malformed = dict(state)
    malformed["lora_A.weight"] = torch.zeros((4, 3))
    with pytest.raises(RuntimeError, match="shape errors"):
        load_lora_state_checked(module, malformed)


def test_calibration_artifact_binds_checkpoint_and_data(tmp_path):
    weight_file = tmp_path / "weights.pt"
    weight_file.write_bytes(b"checkpoint")
    calib_data_file = tmp_path / "calib.json"
    _write_json(calib_data_file, [_sample(f"calibration-{index}") for index in range(20)])
    weight_hash = sha256_file(weight_file)
    data_hash = sha256_file(calib_data_file)
    artifact_file = tmp_path / "conformal_calibration.json"
    _write_json(
        artifact_file,
        {
            "alpha": 0.05,
            "quantile_threshold": 0.2,
            "num_calib_samples": 20,
            "calibration_scores": [0.2] * 20,
            "checkpoint_sha256": weight_hash,
            "metadata": {
                "calibrated_split": str(calib_data_file),
                "calibration_data_sha256": data_hash,
            },
        },
    )

    calibrator, loaded, loaded_data_hash = load_calibration_artifact(
        artifact_file,
        weight_file,
        calib_data_file,
    )
    assert calibrator.quantile_threshold == 0.2
    assert loaded["checkpoint_sha256"] == weight_hash
    assert loaded_data_hash == data_hash

    artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
    del artifact["metadata"]["calibration_data_sha256"]
    _write_json(artifact_file, artifact)
    with pytest.raises(RuntimeError, match="calibration_data_sha256"):
        load_calibration_artifact(artifact_file, weight_file, calib_data_file)


def test_malformed_serialized_artifacts_fail_closed(tmp_path):
    weight_file = tmp_path / "weights.pt"
    weight_file.write_bytes(b"checkpoint")
    calib_data_file = tmp_path / "calib.json"
    _write_json(calib_data_file, [_sample("calibration")])
    artifact_file = tmp_path / "conformal_calibration.json"
    artifact_file.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Calibration artifact must be a JSON/object mapping"):
        load_calibration_artifact(artifact_file, weight_file, calib_data_file)

    with pytest.raises(RuntimeError, match="Checkpoint must be a JSON/object mapping"):
        validate_checkpoint_training([], calib_data_file)


def test_v6_calibration_matches_disjoint_data_and_rejects_legacy_data():
    root = Path(__file__).resolve().parent.parent
    weight_file = root / "output" / "s1_model_v6" / "s1_decision_weights.pt"
    artifact_file = root / "output" / "s1_model_v6" / "conformal_calibration.json"
    disjoint_calib_file = root / "data" / "disjoint_v6" / "calib_v6_disjoint.json"
    legacy_calib_file = root / "data" / "calib_v6.json"
    if not weight_file.exists() or not artifact_file.exists() or not disjoint_calib_file.exists():
        pytest.skip("V6 artifacts are not present")

    # 1. Matches disjoint split correctly
    calibrator, loaded, loaded_hash = load_calibration_artifact(
        artifact_file, weight_file, disjoint_calib_file
    )
    assert calibrator.quantile_threshold > 0.0

    # 2. Rejects legacy non-disjoint split
    if legacy_calib_file.exists():
        with pytest.raises(RuntimeError, match="Calibration split mismatch|calibration data hash mismatch"):
            load_calibration_artifact(artifact_file, weight_file, legacy_calib_file)
