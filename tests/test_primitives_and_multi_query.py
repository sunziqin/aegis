# -*- coding: utf-8 -*-
"""
Unit tests and integration benchmarks for Aegis-S1 primitives and parallel multi-query forward pass.
Verifies full alignment with TypeSafe Jev & Laya specifications.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import open_s1 as s1
from transformers import AutoTokenizer
from open_s1.primitives import Choice, Score, Noul, Boolean, ChoiceResult, ScoreResult, NoulResult
from src.tokenizer_utils import encode_multi_query_batch


BASE_MODEL = __import__("os").environ.get(
    "BASE_MODEL_PATH", "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
)


def _legacy_v6_artifact() -> bool:
    weight_file = Path(__file__).resolve().parent.parent / "output" / "s1_model_v6" / "s1_decision_weights.pt"
    if not weight_file.exists():
        return False
    checkpoint = torch.load(weight_file, map_location="cpu")
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    return not isinstance(config, dict) or not config.get("train_file") or not config.get("train_data_sha256")


def test_primitives_construction():
    """Verify validation and default properties of Choice, Score, and Noul."""
    # Choice
    c = Choice(["A", "B", "C"], question="Which option?")
    assert len(c.options) == 3
    assert c.question == "Which option?"
    
    try:
        Choice(["OnlyOne"])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Score
    s = Score(min_val=1.0, max_val=5.0, steps=5)
    assert len(s.values) == 5
    assert s.values[0] == 1.0
    assert s.values[-1] == 5.0
    assert len(s.options) == 5

    # Noul / Boolean
    n = Noul(question="Is this urgent?")
    assert len(n.options) == 2
    assert n.options[1] == "成立/是/同意"


def test_multi_query_option_spans_stop_at_query_boundary():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    queries = {
        "first": ("Choose the first action", ["A", "B"]),
        "second": ("Choose the second action", ["C", "D"]),
    }
    batch = encode_multi_query_batch(tokenizer, ["state"], [queries], max_length=256)
    input_ids = batch["input_ids"][0]
    first_start, first_end = batch["query_markers"]["first"]["spans"][0, 1].tolist()
    first_text = tokenizer.decode(input_ids[first_start:first_end], skip_special_tokens=True)
    assert "B" in first_text
    assert "[决策指令: second]" not in first_text


def test_live_single_forward_multi_query():
    """Verify single-forward parallel evaluation of multiple queries."""
    model_dir = Path(__file__).resolve().parent.parent / "output" / "s1_model_v6"
    try:
        router = s1.load(model_dir=model_dir)
    except RuntimeError as exc:
        if _legacy_v6_artifact():
            pytest.skip("V6 integration artifact requires retraining with current split provenance")
        raise

    user_state = (
        "用户：我刚才下单的订单 20260921-9981 怎么被系统无故取消了？我付了钱的！"
        "马上给我查清楚，不然我投诉到工信部！"
    )

    schema = {
        "intent": Choice(
            ["query_order_status", "refund_request", "complaint_escalation", "product_consulting"],
            question="用户的主要业务意图是什么？",
        ),
        "urgency": Score(
            min_val=1.0,
            max_val=5.0,
            steps=5,
            labels=["极低", "低", "中", "高", "极端紧急"],
            question="评估用户情绪与诉求的紧急程度",
        ),
        "legal_threat": Noul(
            threshold=0.5,
            question="用户是否存在明确的监管部门投诉威胁？",
        ),
        "requires_human": Noul(
            threshold=0.5,
            question="当前工单是否需要立即转接人工专家处理？",
        ),
    }

    res = router.evaluate(state=user_state, schema=schema, alpha=0.05)

    print("\n" + "=" * 60)
    print("  MULTI-QUERY SINGLE FORWARD PASS RESULTS (Jev/Laya Alignment)")
    print("=" * 60)
    print(f"Overall Verdict: {res.overall_verdict}")
    print(f"Escalate Risk:   {res.escalate_risk:.4f}")
    print(f"Forward Latency: {res.latency_ms:.2f} ms")
    print("-" * 60)

    # 1. Check intent (Choice)
    assert isinstance(res.intent, ChoiceResult)
    print(f"[intent] -> {res.intent.selected_option} (conf: {res.intent.confidence:.4f}, verdict: {res.intent.verdict})")
    print(f"         Prediction set: {res.intent.prediction_set}")

    # 2. Check urgency (Score)
    assert isinstance(res.urgency, ScoreResult)
    print(f"[urgency] -> {res.urgency.score:.2f} / 5.0 (std: {res.urgency.std:.2f}, verdict: {res.urgency.verdict})")

    # 3. Check legal_threat (Noul)
    assert isinstance(res.legal_threat, NoulResult)
    print(f"[legal_threat] -> {res.legal_threat.value} (P(true)={res.legal_threat.probability:.4f}, verdict: {res.legal_threat.verdict})")

    # 4. Check requires_human (Noul)
    assert isinstance(res.requires_human, NoulResult)
    print(f"[requires_human] -> {res.requires_human.value} (P(true)={res.requires_human.probability:.4f}, verdict: {res.requires_human.verdict})")
    print("=" * 60)

    # Assertions
    assert res.latency_ms > 0
    assert len(res.keys()) == 4
    assert res["intent"].selected_option in schema["intent"].options
    assert 1.0 <= res["urgency"].score <= 5.0
    assert hasattr(res, "can_act"), "EvaluationResult must have can_act property"
    assert res.can_act == (res.overall_verdict == "act")
    assert res.is_act() == (res.overall_verdict == "act")
    assert res.intent.can_act == (res.intent.verdict == "act")

    # Benchmark warm latency across 5 runs
    # Warmup
    _ = router.evaluate(state=user_state, schema=schema)
    latencies = []
    for _ in range(5):
        bench_res = router.evaluate(state=user_state, schema=schema)
        latencies.append(bench_res.latency_ms)
    avg_latency = sum(latencies) / len(latencies)
    print(f"Warm Parallel Multi-Query Latency (Avg of 5 runs): {avg_latency:.2f} ms")




if __name__ == "__main__":
    test_primitives_construction()
    print("[PASS] Primitives construction test passed.")
    test_live_single_forward_multi_query()
    print("[PASS] Live single-forward multi-query test passed.")
