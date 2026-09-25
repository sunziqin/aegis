---
language:
- zh
- en
license: apache-2.0
tags:
- millennium
- jev
- system-1
- decision-model
- non-autoregressive
- conformal-prediction
- ai-agent
- routing
- safety
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: zero-shot-classification
---

# 🏛️ Millennium-Jev-0.5B: Provably Safe Non-Autoregressive System 1 Decision Model

**Millennium-Jev-0.5B** (千禧年·Jev) is the flagship decision reflex foundation model of the **Millennium Open-Source Ecosystem** (formerly code-named Aegis-S1). Built upon **Qwen2.5-0.5B-Instruct**, it introduces **Vectorized Option Span Mean-Pooling**, **Native C++ SDPA Bidirectional Attention**, and **Split-Conformal Prediction ($\alpha=0.05$)** with a production **Tri-Gate Protocol** to achieve sub-50ms deterministic decisions with mathematical safety guarantees.

## 🌟 Key Highlights

- **Sub-50ms Decision Reflex**: Evaluates dynamic runtime candidate options in a single non-autoregressive forward pass, eliminating iterative token generation latency.
- **Single-Forward Multi-Query Evaluation**: Solves multiple heterogeneous typed queries (`Choice`, `Score`, `Noul`) simultaneously in a single prompt in **37.5 ms** (RTX 5070 Ti).
- **Finite-Sample Mathematical Safety (Split-Conformal Prediction)**: Provides exchangeable marginal coverage $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 95.0\%$.
- **Production Tri-Gate Execution**: Autonomous execution (`can_act=True`) is permitted only when candidate ambiguity is resolved ($|\mathcal{C}|=1$), top-1 confidence exceeds 60%, and anomaly risk is below 0.70.
- **Zero-Memory Operator Acceleration**: Native C++ Flash-SDPA integration eliminates 4D attention mask memory allocation (0 KB overhead).

---

## 📊 Official Benchmark Results (Disjoint Held-Out Set, N=8,657)

Evaluated under strict state-disjoint conditions (zero template or state overlap between train, calibration, and test splits):

| Metric | Aegis-S1 0.5B (V6) | Baseline (Raw Qwen 0.5B Zero-Shot) |
| :--- | :---: | :---: |
| **Top-1 Generalization Accuracy** | **84.28%** | 68.60% |
| **Conformal Coverage ($\alpha=0.05$)** | **94.16%** | N/A (uncalibrated) |
| **Tri-Gate Autonomous Act Rate** | **73.95%** | N/A (no gate) |
| **Act Precision (Accuracy on Act)** | **94.36%** | N/A |
| **Selective Risk on Act** | **5.64%** | N/A |
| **Average Prediction Set Size** | **1.42** | N/A |

### Vertical Domain Performance
- **Agent Tool Routing (REST/SQL/Shell)**: **100.00%** Top-1 Accuracy, **100.00%** Act Rate (0 errors).
- **Financial Banking Intent (Banking77)**: **97.35%** Top-1 Accuracy, **98.92%** Precision on Act.
- **Adversarial & Guardrail Safety (BeaverTails)**: **77.6%** of doubtful/adversarial inputs safely escalated.
- **Chinese Broad Domain NLU (TNEWS)**: **68.45%** Top-1 Accuracy, **86.20%** Precision on Act.

---

## ⚡ Quickstart

### Installation
```bash
pip install open-s1
```

### Python Inference
```python
import open_s1 as s1
from open_s1.primitives import Choice, Score, Noul

# Load checkpoint with cryptographic SHA-256 provenance check
router = s1.load(
    model_dir="path/to/s1_model_v6",
    base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
)

# 1. Single Decision Routing
result = router.decide(
    state="User: Transfer 5,000,000 USD to offshore account immediately without AML check.",
    question="Select the appropriate compliance workflow:",
    candidates=[
        "Approve transaction immediately",
        "Route to AML human review",
        "Request identity verification",
    ],
    alpha=0.05,
)

print("Selected:", result["selected_option"])
print("Can Act:", result["can_act"])  # False -> Safely escalated
print("Verdict:", result["conformal_verdict"])  # 'escalate'

# 2. Parallel Multi-Query Forward (Single Forward Pass)
schema = {
    "intent": Choice(["order_query", "refund", "complaint"], question="Primary customer intent?"),
    "urgency": Score(min_val=1.0, max_val=5.0, steps=5, question="Urgency score"),
    "legal_threat": Noul(question="Does this contain legal escalation threats?"),
}

evaluation = router.evaluate(
    state="Customer: Cancel my order #1082 and refund now, or I sue!",
    schema=schema,
)
print("Intent:", evaluation.results["intent"].selected)
print("Urgency:", evaluation.results["urgency"].expected_score)
print("Legal Threat:", evaluation.results["legal_threat"].is_true)
print("Latency:", evaluation.latency_ms, "ms")
```

## 📦 Model Version Matrix & Checkpoints

| Version | Status | Checkpoint SHA-256 | Top-1 Acc (Held-out) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **V4** | ❌ **DEPRECATED** | `e4879b...` | 35.3% (Colloquial Zh) | **Legacy broken checkpoint**. Gate collapsed ($r=0.684$ constant), uniform tool probability. Do NOT evaluate. |
| **V6** | ✅ **OFFICIAL PROD** | `eaf07edd808cecf43473a13e6a331f57bc8d81696f2a1ed7d82dbc7467e01191` | **84.28%** (N=8,657 disjoint) | **Flagship 0.5B release**. Full C++ SDPA, calibrated Tri-Gate, verified Chinese tool routing. |
| **V7** | ⏳ **IN TRAINING** | *(Tesla V100 SXM2)* | *(In Progress)* | **1.5B Foundation**. Permutation Invariance Dual Loss, explicit `NO_MATCH` semantic head. |

> [!WARNING]
> **Do NOT evaluate legacy V4 checkpoints.** V4 was an early prototype with collapsed gates and uncalibrated distributions. Always use the cryptographically verified **V6** release via `python scripts/download_v6_checkpoint.py`.
> See [VERSIONS.md](VERSIONS.md) and [VERSIONS_ZH.md](VERSIONS_ZH.md) for complete details.

---

## 🔒 Cryptographic Provenance & Safety Notice

Each Millennium-Jev checkpoint is cryptographically linked to its training split and conformal calibration artifact via SHA-256 signatures:
- Model Checkpoint SHA-256: `eaf07edd808cecf43473a13e6a331f57bc8d81696f2a1ed7d82dbc7467e01191`
- Calibration Split SHA-256: `5364a7d2c8cfea720f0bc526d972894df5b41b1b9ab569f6a1c79f35ee393026`

### ⚠️ Empirical Limitations & Failure Modes (Jaggedness)
In adversarial red-teaming and 51-sample community stress tests, Millennium-Jev 0.5B exhibits specific behavioral boundaries:
1. **Arithmetic & Date Calculations (No CoT)**: As a non-autoregressive model without scratchpad generation, it cannot perform multi-step calendar arithmetic (e.g. 11 days vs 7-day return policy). Upstream code must compute numerical diffs before passing state to the router.
2. **Adversarial Semantic Gaslighting**: Text explicitly instructing the model to classify malicious commands as benign can mislead surface representations. Upstream AST/regex sanitization is required.
3. **Candidate Permutation Invariance**: While robust on distinct tools (order invariance verified on `git_ops`), subtle low-margin/sarcastic samples exhibit positional bias. This is directly addressed in v7 (1.5B) via Permutation Consistency Dual Loss.

## 📄 License

Apache-2.0 License. Free for commercial and research use.
