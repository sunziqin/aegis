# -*- coding: utf-8 -*-
"""
Aegis-S1 State-Disjoint Grouped Dataset Partitioning.
Groups all 59,548+ samples across domains by unique state hash,
guaranteeing 100% zero-overlap across Train, Calibration, and Test splits:
State(Train) ∩ State(Calib) = ∅
State(Train) ∩ State(Test)  = ∅
State(Calib) ∩ State(Test)  = ∅
"""

import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def normalize_state(text: str) -> str:
    return " ".join(text.strip().lower().split())


def state_hash(text: str) -> str:
    return hashlib.md5(normalize_state(text).encode("utf-8")).hexdigest()


def partition_strictly_disjoint(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    calib_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 2026,
):
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all samples
    all_samples = []
    for fname in ["train_v6.json", "calib_v6.json", "test_v6.json"]:
        p = data_dir / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_samples.extend(data)
                logger.info(f"Loaded {len(data):,} samples from {p.name}")

    total_samples = len(all_samples)
    logger.info(f"Total Combined Pool: {total_samples:,} samples")

    # 2. Group by normalized state hash to prevent leakage
    state_to_samples = defaultdict(list)
    for s in all_samples:
        h = state_hash(s["state"])
        state_to_samples[h].append(s)

    unique_hashes = list(state_to_samples.keys())
    rng.shuffle(unique_hashes)
    logger.info(f"Unique States Identified: {len(unique_hashes):,} (Average {total_samples / len(unique_hashes):.2f} samples per state)")

    # 3. Partition unique state hashes
    n_unique = len(unique_hashes)
    n_train_h = int(n_unique * train_ratio)
    n_calib_h = int(n_unique * calib_ratio)

    # Keep the shuffled list order. Iterating a set here makes the output
    # depend on PYTHONHASHSEED, so identical seeds would produce different
    # JSON files across processes.
    train_hashes = unique_hashes[:n_train_h]
    calib_hashes = unique_hashes[n_train_h : n_train_h + n_calib_h]
    test_hashes = unique_hashes[n_train_h + n_calib_h :]
    train_hash_set = set(train_hashes)
    calib_hash_set = set(calib_hashes)
    test_hash_set = set(test_hashes)

    # 4. Verify zero overlap of state hashes
    assert len(train_hash_set.intersection(calib_hash_set)) == 0, "Train and Calib leak detected!"
    assert len(train_hash_set.intersection(test_hash_set)) == 0, "Train and Test leak detected!"
    assert len(calib_hash_set.intersection(test_hash_set)) == 0, "Calib and Test leak detected!"

    train_samples = []
    calib_samples = []
    test_samples = []

    for h in train_hashes:
        train_samples.extend(state_to_samples[h])
    for h in calib_hashes:
        calib_samples.extend(state_to_samples[h])
    for h in test_hashes:
        test_samples.extend(state_to_samples[h])

    rng.shuffle(train_samples)
    rng.shuffle(calib_samples)
    rng.shuffle(test_samples)

    # 5. Verify zero overlap of raw state strings
    tr_states = set(s["state"] for s in train_samples)
    ca_states = set(s["state"] for s in calib_samples)
    te_states = set(s["state"] for s in test_samples)

    tr_ca_overlap = tr_states.intersection(ca_states)
    tr_te_overlap = tr_states.intersection(te_states)
    ca_te_overlap = ca_states.intersection(te_states)

    logger.info("\n" + "=" * 70)
    logger.info("   STATE-DISJOINT PARTITION VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"  Train Samples:      {len(train_samples):,} ({len(train_samples)/total_samples:.1%}) | Unique States: {len(tr_states):,}")
    logger.info(f"  Calib Samples:      {len(calib_samples):,} ({len(calib_samples)/total_samples:.1%}) | Unique States: {len(ca_states):,}")
    logger.info(f"  Test Samples:       {len(test_samples):,} ({len(test_samples)/total_samples:.1%}) | Unique States: {len(te_states):,}")
    logger.info(f"  Overlap Train-Calib: {len(tr_ca_overlap)} (Target: 0)")
    logger.info(f"  Overlap Train-Test:  {len(tr_te_overlap)} (Target: 0)")
    logger.info(f"  Overlap Calib-Test:  {len(ca_te_overlap)} (Target: 0)")
    logger.info("=" * 70)

    assert len(tr_ca_overlap) == 0
    assert len(tr_te_overlap) == 0
    assert len(ca_te_overlap) == 0
    logger.info("[SUCCESS] 100% Mathematical Zero Overlap Verified!")

    # 6. Save repartitioned splits
    out_train = output_dir / "train_v6_disjoint.json"
    out_calib = output_dir / "calib_v6_disjoint.json"
    out_test = output_dir / "test_v6_disjoint.json"

    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False)
    with open(out_calib, "w", encoding="utf-8") as f:
        json.dump(calib_samples, f, ensure_ascii=False)
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False)

    logger.info(f"Saved: {out_train.name}, {out_calib.name}, {out_test.name}")


if __name__ == "__main__":
    data_dir = Path("E:/s1-decision-model/data")
    output_dir = Path("E:/s1-decision-model/data/disjoint_v6")
    partition_strictly_disjoint(data_dir, output_dir)
