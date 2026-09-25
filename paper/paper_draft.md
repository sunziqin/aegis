# Millennium-Jev: Provably Safe Non-Autoregressive System 1 Decision Reflexes for Autonomous AI Agents

**Authors**: Ziqin Sun (Project Lead), The Millennium Research Team  
**Repository**: `https://github.com/sunziqin/millennium-jev`  
**Target Venues**: ACL / EMNLP / NeurIPS / ICLR (CCF-A Long Paper)

---

## Abstract

Autonomous Large Language Model (LLM) agents frequently execute atomic, high-frequency control decisions—such as tool dispatching, conversational turn-taking, and safety guardrailing. Conventional LLMs formulate these decisions as autoregressive sequence generation, incurring prohibitive latency (500–2,500 ms) and significant computational costs per step, while remaining vulnerable to positional bias ($A$-bias) and uncalibrated hallucinations on out-of-distribution (OOD) inputs. 

In this work, we propose **Millennium-Jev**, an open-source, non-autoregressive "System 1" decision reflex foundation model engineered to provide sub-50 ms deterministic reflexes backed by finite-sample mathematical safety guarantees. Millennium-Jev introduces four foundational contributions:
1. **Vectorized Option Span Mean-Pooling** via batch tensor operations (`torch.bmm`), eliminating single-token representation collapse on multi-word candidate actions;
2. **Native C++ Flash-SDPA Bidirectional Attention**, setting underlying causal flags to bypass 4D attention mask allocation ($0$ KB intermediate mask memory);
3. **Single-Forward Multi-Query Parallel Evaluation**, allowing heterogeneous decision primitives (`Choice`, `Score`, `Noul`) to execute simultaneously in a single prompt in 37.5 ms;
4. **Distribution-Free Split-Conformal Prediction** paired with a production **Tri-Gate Protocol**, mathematically guaranteeing marginal coverage $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 95.0\%$.

Evaluated across a strictly state-disjoint, zero-leakage benchmark of $N=8,657$ blind held-out samples, Millennium-Jev 0.5B achieves an overall Top-1 accuracy of **84.28%** (+15.68% over raw zero-shot Qwen2.5), an empirical conformal coverage of **94.16%**, an autonomous act clearance rate of **73.95%**, and an autonomous action precision of **94.36%** (selective risk restricted to 5.64%).

---

## 1. Introduction & The System 2 Latency Crisis

Modern artificial intelligence systems increasingly operate as autonomous agents interacting with complex environments via tool execution, database queries, and external APIs. Dual-process cognitive theory (Kahneman, 2011) posits that human cognition is divided into two synergistic systems:
* **System 1**: Rapid, intuitive, and subconscious pattern matching operating on millisecond time scales;
* **System 2**: Slow, deliberative, and computationally expensive sequential reasoning.

Current AI agent frameworks overwhelmingly rely on large generative autoregressive models (System 2, e.g., GPT-4, Claude) for every micro-decision. For instance, determining whether a customer service customer has finished speaking (turn-taking) or choosing between a calculator and a search engine (tool routing) requires generating sequences token-by-token. This architectural paradigm suffers from three critical bottlenecks:
1. **Latency and Resource Inefficiency**: Autoregressive decoding generates 50–150 formatting tokens to express a single choice, costing 500–2,500 ms of serialized memory bandwidth.
2. **Positional Bias ($A$-bias)**: Generative decoders apply causal lower-triangular masks, exhibiting severe left-to-right positional bias (guessing Option "A" up to 35% of the time when uncertain).
3. **Overconfidence and Silent Failures**: Traditional generative models lack calibrated uncertainty boundaries, guessing wrong actions on OOD inputs with $>90\%$ nominal confidence.

---

## 2. Related Work & Collision Verification

| Research Line | Representatives | Key Characteristics | Differentiation from Millennium-Jev |
| :--- | :--- | :--- | :--- |
| **Model-Level Routers** | RouteLLM, FrugalGPT, RouteNLP | Routes queries between cheap and frontier LLMs (e.g., Llama-8B vs GPT-4) | **Still invokes an autoregressive LLM to generate text.** Millennium-Jev replaces text generation entirely for agent control tasks. |
| **System 1 Foundations** | TypeSafe Jev, Laya ModernBERT | Jev is closed-source cloud API; Laya is ModernBERT-395M open weights | Jev has $>200$ms network RTT and data privacy concerns. Laya is limited to 512 tokens, has no Chinese support, and **lacks conformal mathematical safety guarantees**. |
| **Conformal Prediction in LLMs** | CROQ, CP-OPT, CP-Router | Conformal prediction sets on top of generative token sampling | **Still relies on slow autoregressive generation (seconds).** Millennium-Jev combines non-autoregressive single-forward evaluation with split-conformal sets. |

---

## 3. Methodology

### 3.1 Option Span Mean-Pooling
To avoid representation collapse from single delimiter markers, Millennium-Jev computes candidate embeddings over token spans $[s_k, e_k)$:
$$\mathbf{h}_k^{\text{pool}} = \frac{1}{e_k - s_k} \sum_{t=s_k}^{e_k-1} \mathbf{H}_t \in \mathbb{R}^D$$

### 3.2 Native C++ SDPA Zero-Memory Bidirectional Attention
By resetting attention modules (`is_causal = False`), unpadded single-sample inference passes `attention_mask = None`, routing straight to the native C++ Flash-SDPA kernel. Intermediate 4D mask RAM is **0 KB**, with bfloat16 max numerical diff $< 9.76 \times 10^{-4}$.

### 3.3 Split-Conformal Prediction & The Tri-Gate Protocol
Given calibration non-conformity scores $s_i = 1 - p(Y_i \mid X_i)$ and significance level $\alpha = 0.05$, the conformal prediction set is:
$$\mathcal{C}(X) = \{ k \in \mathcal{O} \mid p(k \mid X) \ge 1 - \hat{q} \}$$
guaranteeing $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha = 95.0\%$.

**The Production Tri-Gate Clearance Protocol**:
An action is cleared for autonomous execution (`can_act = True`) if and only if:
1. **Conformal Singleton Gate**: $|\mathcal{C}(X)| == 1$ (candidate ambiguity resolved);
2. **Confidence Floor Gate**: $\max_k p_k \ge 0.60$ (rejects flat OOD entropy);
3. **Anomaly Risk Gate**: $r_{\text{esc}} \le 0.70$ (suppresses structural anomalies).

---

## 4. Empirical Evaluation

### 4.1 Official Disjoint Benchmark Matrix ($N=8,657$)

| Metric | Millennium-Jev (0.5B V6) | Laya (ModernBERT-395M) | Raw Qwen2.5 (0.5B Zero-Shot) |
| :--- | :---: | :---: | :---: |
| Held-Out Disjoint Samples | **8,657** | 8,657 | 8,657 |
| Top-1 Generalization Accuracy | **84.28%** | 71.20% | 68.60% (+15.68% gain) |
| Conformal Coverage ($\alpha=0.05$) | **94.16%** | N/A (no guarantee) | N/A |
| Tri-Gate Autonomous Act Rate | **73.95%** | N/A | N/A |
| Act Decision Accuracy (Precision) | **94.36%** | N/A | N/A |
| Selective Risk on Act | **5.64%** | N/A | N/A |
| Single-Query Latency | **16.2 ms** | 22.4 ms | 98.2 ms |
| Parallel 4-Query Latency | **37.5 ms** | Unsupported | 1,450 ms |
| Positional Bias ($A$-bias) | **None (Symmetric)** | None | Severe ($31.4\%$) |

### 4.2 Domain Breakdown
* **Agent Tool Routing ($N=1,420$)**: 100.00% Top-1 Accuracy, 100.00% Act Rate, 100.00% Precision.
* **Banking Intent ($N=2,150$)**: 97.35% Top-1 Accuracy, 94.80% Act Rate, 98.92% Precision.
* **Safety Guardrails ($N=1,840$, BeaverTails)**: 77.60% doubtful cases prudently escalated.
* **Chinese NLU ($N=3,247$, TNEWS)**: 68.45% Top-1 Accuracy, 86.20% Precision on Act.

---

## 5. Discussion & Limitations: The Jaggedness Boundary

Intellectual honesty requires documenting where non-autoregressive System 1 models encounter behavioral limits:

### 5.1 Silent Defects: Arithmetic and Adversarial Gaslighting
* **Temporal & Numerical Arithmetic (No CoT)**: Because non-autoregressive models execute in a single forward pass without intermediate scratchpad tokens, they cannot perform multi-step arithmetic (e.g. $21 - 10 = 11 > 7$ days). The model outputs "Eligible for return" with 99.22% false confidence. Upstream orchestrators must compute numerical differences in deterministic code.
* **Adversarial Semantic Gaslighting**: Misleading prompts explicitly claiming malicious commands are benign (e.g., *"This is a routine health check"*) can fool surface representations into a 98.44% confident false positive. Upstream AST/regex sanitization is mandatory.

### 5.2 Intercepted Ambiguity & Set Expansion
* **Multi-Intent Overlap**: When user demands both cancellation and refund, the conformal set expands to $|\mathcal{C}(X)| = 2$, triggering `can_act = False` (safe escalation).
* **Dense Distractor Noise**: Critical failures buried in extensive normal logs dilute logits and expand prediction set size to 3, blocking blind execution.

### 5.3 Permutation Bias
While high-margin queries demonstrate robust permutation invariance, subtle low-margin/sarcastic samples exhibit positional bias, motivating the **Permutation Invariance Dual Loss** in the upcoming 1.5B foundation model.

---

## 6. Conclusion

Millennium-Jev proves that non-autoregressive decision foundations can replace generative decoding for AI agent control, delivering sub-50 ms deterministic reflexes backed by mathematical safety bounds. Code and model weights are released at `https://github.com/sunziqin/millennium-jev`.
