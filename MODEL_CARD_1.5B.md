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
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: zero-shot-classification
---

# 🏛️ Millennium-Jev-1.5B: Flagship Non-Autoregressive System 1 Decision Model

**Millennium-Jev-1.5B** (千禧年·Jev 旗舰版) is the flagship decision reflex model of the **Millennium Open-Source Ecosystem** (formerly code-named Aegis-S1). Built upon **Qwen2.5-1.5B-Instruct**, it features **1.54B backbone parameters** coupled with **77.08M trainable decision parameters** (36.93M LoRA adapter + 40.15M non-autoregressive Cross-Attention decision head).

With **Vectorized Option Span Mean-Pooling**, **Native C++ SDPA Bidirectional Attention**, and **Split-Conformal Prediction ($\alpha=0.05$)**, Millennium-Jev-1.5B delivers sub-50ms deterministic decisions with tighter uncertainty bounds, higher autonomous execution rates, and rigorous mathematical safety guarantees.

## 🌟 Key Highlights

- **Enhanced Autonomous Act Rate**: Achieves **74.54%** autonomous decision pass rate on state-disjoint test sets (+2.40% over the 0.5B baseline) under strict $1-\alpha=95\%$ safety bounds.
- **Lower Selective Risk**: Selective risk on autonomous action dropped to **5.52%** (an **11% relative reduction in errors** compared to 0.5B's 6.22%).
- **Sub-50ms Decision Reflex**: Evaluates dynamic runtime candidate options in a single non-autoregressive forward pass, eliminating iterative autoregressive token generation latency.
- **Single-Forward Multi-Query Evaluation**: Solves multiple heterogeneous typed queries (`Choice`, `Score`, `Noul`) simultaneously in a single prompt.
- **Finite-Sample Mathematical Safety (Split-Conformal Prediction)**: Provides exchangeable marginal coverage $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 94.09\%$ with average prediction set size of **1.366**.
- **Cryptographic Provenance Verification**: Bound to SHA-256 checkpoint verification (`458c3b73...`) preventing silent fallback to uncalibrated legacy weights.

---

## 📊 Official Benchmark Results (Disjoint Held-Out Set, N=8,657)

Evaluated under strict state-disjoint conditions (zero template or state overlap between train, calibration, and test splits):

| Metric | Millennium-Jev 1.5B (Flagship) | Millennium-Jev 0.5B (V6) | Baseline (Raw Qwen Zero-Shot) |
| :--- | :---: | :---: | :---: |
| **Top-1 Generalization Accuracy** | **84.88%** | 84.28% | 68.60% |
| **Conformal Coverage ($\alpha=0.05$)** | **94.09%** | 94.16% | N/A (uncalibrated) |
| **Tri-Gate Autonomous Act Rate** | **74.54%** | 72.14% | N/A (no gate) |
| **Selective Risk on Act** | **5.52%** | 6.22% | N/A |
| **Abstention / Escalation Rate** | **25.46%** | 27.86% | N/A |
| **Average Prediction Set Size** | **1.366** | 1.390 | N/A |

### Vertical Domain Performance Breakdown
- **Agent Tool Routing (REST/SQL/Shell)**: **100.00%** Top-1 Accuracy, **100.00%** Act Rate (0 errors).
- **High-Cardinality Actions (20+ candidates)**: **100.00%** Top-1 Accuracy, **99.93%** Act Rate.
- **Financial Banking Intent (Banking77)**: **97.38%** Top-1 Accuracy, **96.38%** Autonomous Act Rate.
- **Chinese Broad Domain NLU (TNEWS 15-class)**: **68.03%** Top-1 Accuracy (+2.79% over 0.5B), **59.92%** Act Rate (+7.91%).
- **Adversarial & Guardrail Safety (BeaverTails)**: **68.71%** Top-1 Accuracy, **28.70%** Act Rate (+9.39%), safely escalating ambiguous/adversarial attacks.

---

## ⚡ Quickstart

### Installation
```bash
pip install open-s1
```

### Python Inference with 1.5B Flagship
```python
import open_s1 as s1
from open_s1.primitives import Choice, Score, Noul

# Load 1.5B Flagship model
router = s1.load(
    model_dir="sunziqin/millennium-jev-1.5b",
    base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
)

# 1. Single Decision Routing with Conformal Risk Tri-Gate
result = router.decide(
    state="System alert: Disk IO latency spiked to 450ms on database node replica-03.",
    question="Select automated remediation action:",
    candidates=[
        "Restart entire database cluster",
        "Drain node replica-03 and failover to standby",
        "Ignore alert as transient noise",
    ],
    alpha=0.05,
)

print("Selected:", result["selected_option"])
print("Can Act:", result["can_act"])            # True
print("Confidence:", result["confidence"])      # e.g., 0.984
print("Prediction Set:", result["conformal_set"])

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
print("Legal Threat:", evaluation.results["legal_threat"].value)
```

---

## 🔒 Cryptographic Checkpoint Provenance

```json
{
  "checkpoint_file": "s1_decision_weights.pt",
  "checkpoint_sha256": "458c3b73157e05a99ae53b0a2e1eb090b3d434bc5facfb482b1fd884968f177d",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "conformal_q_hat": 0.95361328125,
  "nominal_coverage": 0.95,
  "calibration_samples": 10075,
  "test_samples": 8657,
  "license": "Apache-2.0"
}
```

## 📜 Citation

If you use Millennium-Jev in your research or production systems, please cite:

```bibtex
@software{millennium_jev_2026,
  author = {Ziqin Sun},
  title = {Millennium-Jev: Provably Safe Non-Autoregressive System 1 Decision Model},
  year = {2026},
  url = {https://github.com/sunziqin/millennium-jev}
}
```
