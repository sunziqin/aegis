# Aegis-S1: Provably Safe Non-Autoregressive System-1 Decision Models for LLM Agents

**Authors**: Anonymous Authors  
**Target Venue**: ACL / EMNLP / NeurIPS (2026/2027)

---

## Abstract

Autonomous Large Language Model (LLM) agents frequently execute atomic, high-frequency control decisions—such as tool dispatching, conversational turn-taking, and safety guardrailing. Conventional LLMs formulate these decisions as autoregressive sequence generation, incurring prohibitive latency (100–1000 ms) and significant computational costs per step, while remaining vulnerable to decoding hallucinations and severe positional biases. In this work, we propose **Aegis-S1**, an open-source, non-autoregressive "System 1" decision framework that eliminates token-by-token generation in favor of a single forward pass. 

Aegis-S1 introduces:
1. A **Dynamic Option-Marker Attention Head** that models mutual competition across variable candidate actions;
2. **Distribution-free Conformal Prediction** calibration, establishing rigorous, finite-sample statistical safety guarantees ($P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$) that convert epistemic uncertainty into principled escalation rather than silent failures;
3. A reference implementation with a 4096-token hard input limit; its bidirectional 4D attention has $O(L^2)$ memory cost, so longer-context and bilingual behavior require separate measurement.

The current repository does not contain a regenerated, provenance-validated V6 result. The historical artifacts are intentionally rejected by the evaluator and SDK, so accuracy, latency, and OOD safety numbers must be regenerated and reported from the current disjoint splits before they are used as empirical claims.

---

## 1. Introduction

Modern artificial intelligence systems increasingly operate as autonomous agents interacting with complex environments via tool execution, database queries, and external APIs (Schick et al., 2023; Yao et al., 2022). Dual-process cognitive theory (Kahneman, 2011) posits that human cognition is divided into two synergistic systems:
* **System 1**: Rapid, intuitive, and subconscious pattern matching operating on millisecond time scales;
* **System 2**: Slow, deliberative, and computationally expensive sequential reasoning.

Current AI agent frameworks overwhelmingly rely on large generative autoregressive models (System 2, e.g., GPT-4, Claude) for every micro-decision. For instance, determining whether a customer service customer has finished speaking (turn-taking) or choosing between a calculator and a search engine (tool routing) requires generating sequences token-by-token. This architectural paradigm suffers from three critical bottlenecks:

1. **Latency and Resource Inefficiency**: Autoregressive decoding requires sequential memory bandwidth lookups against large Key-Value (KV) caches, resulting in 100–500 ms overheads even for small models.
2. **Positional Bias and Formatting Fragility**: Generative decoders exhibit pronounced left-to-right positional bias (e.g., an inherent preference for Option "A" when uncertain) and are prone to syntax parsing failures.
3. **Overconfidence and Silent Failures**: Traditional generative models lack calibrated uncertainty boundaries, frequently guessing wrong tools with high nominal confidence.

Recently, closed-source models such as TypeSafe Jev and open-source models such as Laya (2026) have attempted to revive encoder-based decision systems. However, existing implementations suffer from severe limitations: Laya truncates the context window to 512 tokens, relies on heuristic reinforcement learning (RLCD) without theoretical safety guarantees, and is restricted to English.

In this paper, we introduce **Aegis-S1**, addressing these limitations through three foundational contributions:
* **Architectural Scaling & Invariance**: We design a dynamic option-marker scoring mechanism with inter-option cross-attention. The current reference implementation supports up to 4096 input tokens and must be benchmarked for memory at each sequence length.
* **Provable Epistemic Safety**: We incorporate Split Conformal Prediction into the System 1 decision pipeline, establishing a mathematical upper bound on error rates ($\alpha \le 5\%$) that enables provably safe automated execution and principled escalation.
* **Empirical Validation**: We provide disjoint train/calibration/test tooling and provenance-bound reports. Quantitative claims remain pending until a current V6 checkpoint is retrained, calibrated, and evaluated.

---

## 2. Methodology

```
Input Tokens: [State] ... [Question] ... [MARKER] Opt_1 ... [MARKER] Opt_K
                            │
              ▼───────────────────────────▼
              [ 24-Layer Transformer Encoder ] (Native Bidirectional / Prefix)
                            │
              ▼───────────────────────────▼
              [ Extract Marker Embeddings: H_opt ∈ R^{K x D} ]
                            │
              ▼───────────────────────────▼
              [ 2-Layer Inter-Option Transformer ] (Mutual Competition)
                            │
              ▼───────────────────────────▼
              [ Linear Projection + Masked Softmax ]
                            │
              ▼───────────────────────────▼
              [ Conformal Risk Filter: C(x) ⊆ {1...K} ]
                 /                         \
       |C(x)| = 1                      |C(x)| ≠ 1
          │                                │
     [ Automated Act ]            [ Safe Escalate / Reject ]
```

### 2.1 Dynamic Option-Marker Architecture

Given a decision context $\mathbf{S}$ (state and instructions) and a dynamic set of $K$ candidate actions $\mathcal{O} = \{o_1, o_2, \dots, o_K\}$, we format the input as a unified token sequence:
$$\mathbf{X} = [\text{CLS}] \circ \mathbf{S} \circ \bigoplus_{k=1}^K \left( \tau_{\text{marker}} \circ o_k \right) \circ [\text{SEP}]$$
where $\tau_{\text{marker}}$ is a dedicated marker token (e.g., `[MASK]` or `<|fim_pad|>`), and $K$ can vary dynamically at inference time ($K \in [2, K_{\max}]$).

Let $\mathbf{H} \in \mathbb{R}^{L \times D}$ be the hidden representations produced by the transformer backbone. We locate the token coordinates of each candidate marker $m_1, m_2, \dots, m_K$ and gather their embeddings:
$$\mathbf{E}_{\text{cand}} = [\mathbf{h}_{m_1}, \mathbf{h}_{m_2}, \dots, \mathbf{h}_{m_K}]^\top \in \mathbb{R}^{K \times D}$$

### 2.2 Inter-Option Cross-Attention

To enforce symmetric competition and mutual exclusivity among candidate choices without positional bias, $\mathbf{E}_{\text{cand}}$ is passed through a 2-layer Transformer encoder:
$$\widetilde{\mathbf{E}}_{\text{cand}} = \text{TransformerLayer}(\mathbf{E}_{\text{cand}}, \text{Mask}_{\text{cand}})$$
A linear projection layer computes raw candidate logits:
$$z_k = \mathbf{w}_{\text{score}}^\top \widetilde{\mathbf{e}}_k + b$$
and the choice probability distribution is obtained via masked softmax:
$$p_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$$

### 2.3 Strictly Proper Calibration Loss

To mitigate the overconfidence inherent in standard Cross-Entropy ($\mathcal{L}_{\text{CE}}$), we train the network under a compound calibrated loss incorporating the **Brier Score** (a strictly proper scoring rule):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{Brier}} + \lambda_2 \mathcal{L}_{\text{Escalate}}$$
where:
$$\mathcal{L}_{\text{Brier}} = \frac{1}{K} \sum_{k=1}^K (p_k - y_k)^2, \quad y_k \in \{0, 1\}$$
Mathematical theory guarantees that the Brier score is minimized if and only if predicted probabilities match true underlying posterior likelihoods.

### 2.4 Finite-Sample Conformal Safety Guarantees

Rather than emitting a point prediction, Aegis-S1 computes a **prediction set** $\mathcal{C}(X) \subseteq \mathcal{O}$.

**Theorem 1 (Vovk et al., 2005; Romano et al., 2020)**:  
Given exchangeable calibration samples $\{ (X_i, Y_i) \}_{i=1}^n$ and a significance level $\alpha \in (0, 1)$, define non-conformity scores $s_i = 1 - p(Y_i \mid X_i)$. Let $\hat{q}$ be the $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$-th empirical quantile of $\{s_i\}$. Then for any new test point $(X_{n+1}, Y_{n+1})$, the prediction set:
$$\mathcal{C}(X_{n+1}) = \{ k \in \mathcal{O} \mid p(k \mid X_{n+1}) \ge 1 - \hat{q} \}$$
satisfies the marginal coverage guarantee:
$$\mathbb{P}\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \ge 1 - \alpha$$

**Operational Decision Policy**:
* **Automate (`Act`)**: If $|\mathcal{C}(X)| = 1$, execute the single safe candidate.
* **Escalate (`Escalate`)**: If $|\mathcal{C}(X)| > 1$, multiple candidates meet the guarantee; route to System 2 or human-in-the-loop.
* **Refuse (`Reject`)**: If $|\mathcal{C}(X)| = 0$, the input is out-of-distribution; reject execution.

---

## 3. Experimental Evaluation

### 3.1 Benchmark Datasets
We evaluate models across two benchmark suites:
1. **S1-Bench-100**: 103 in-distribution tasks across Agent Tool Routing (45), Guardrails (30), and Dialogue Turn-Taking (28).
2. **S1-OOD-Bench**: 25 held-out OOD-style tasks spanning Bioinformatics, Quantitative Finance, 3D Graphics Shaders, Cloud DevOps, and Steganographic Injections. The current split contract verifies normalized-state disjointness; it does not by itself prove that every question template or domain term is unseen.

### 3.2 In-Distribution Results

The current V6 result table is intentionally pending. The repository contains only legacy artifacts for the earlier benchmark; they are rejected by the current provenance checks and cannot support accuracy or latency claims. Regenerate this table from the current disjoint splits and record the dependency versions, checkpoint hash, calibration hash, and target hardware with the report.

### 3.3 Out-of-Distribution (OOD) Safety & Refusal

On the held-out OOD-style probe set, report the coverage, selective risk, act rate, and escalation rate from the regenerated provenance-bound report. Historical OOD numbers are not carried forward into this draft.

---

## 4. Related Work & Discussion
* **Non-Autoregressive Transformers**: Gu et al. (2018), Ghazvininejad et al. (2019).
* **Calibrated Decision Making**: Proper scoring rules (Gneiting & Raftery, 2007), Conformal Risk Control (Angelopoulos & Bates, 2021).
* **LLM Agents & Routing**: Toolformer (Schick et al., 2023), NexusRaven (2024), Laya (2026).

---

## 5. Conclusion
Aegis-S1 provides a non-autoregressive decision path and a conformal escalation mechanism. The coverage guarantee depends on a valid exchangeable calibration protocol; latency, accuracy, and context claims remain empirical questions for the regenerated evaluation.
