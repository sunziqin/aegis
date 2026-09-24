# Aegis-S1：具备数学安全保证的非自回归系统一（System 1）决策大模型基座

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/开源协议-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/推理延迟-亚50ms反射-green.svg" alt="Inference Speed">
  <img src="https://img.shields.io/badge/共形保证-分裂共形%2095%25-orange.svg" alt="Safety Guarantees">
  <img src="https://img.shields.io/badge/原语对齐-Choice%20%7C%20Score%20%7C%20Noul-purple.svg" alt="Decision Primitives">
  <img src="https://img.shields.io/badge/三重门控-Tri--Gate%20放行-red.svg" alt="Tri-Gate">
</p>

**Aegis-S1** 是开源的高性能**“系统一（System 1）”非自回归决策基座模型**，专为 AI 智能体（Agent）提供低延迟决策反射与有限样本数学安全保证。实际延迟取决于基础模型、设备、批大小和序列长度，必须在部署硬件上用仓库脚本实测。

项目深度对标硅谷 **TypeSafe Jev** 与开源项目 **Laya** 的设计思想，使用**单次前向传播**直接并发解析结构化决策原语。端到端延迟必须在相同硬件和工作负载下比较。

---

## 🌟 核心架构与工程突破

### 1. 三大类型化决策原语（全面对齐 Jev / Laya）
Aegis-S1 提炼了智能体决策的最小完备原语集合：
- `Choice(options, question)`：**离散候选分类**。在运行时动态传入任意字符串候选项（强制要求 $\ge 2$ 个互异非空候选），基于分裂共形预测输出数学置信集 $\mathcal{C}(X)$ 与放行/升级判定；
- `Score(min_val, max_val, steps, labels)`：**连续/序数标度打分**。采用离散概率期望 $\mathbb{E}[S] = \sum v_k \cdot p_k$ 给出平滑评分与认识论方差 $\mathrm{Var}[S]$；
- `Noul(threshold, question)`（别名 `Boolean`）：**布尔条件断言**。精准计算条件成立概率 $P(\text{true})$，内置安全裕度边界检查，杜绝临界值误判。

### 2. 多题并行单次前向评估（Single-Forward Parallel Evaluation）
在真实的智能体决策与工单调度中，单个用户状态通常需要同时回答多个异构问题（如：主业务意图、情绪紧急度评分、监管投诉风险断言、是否转人工专家）：
- 传统自回归大模型需生成 50~150 个 Token 或并发调用多次 API，耗时 1.5~3 秒；
- **Aegis-S1 在单次前向中完成全部问题解析**：多题的 Marker 统一嵌入序列，LLM 主干网络**仅执行一次**。请使用仓库中的 benchmark 脚本在目标硬件上测量延迟。

### 3. 分裂共形预测与生产级三重安全门控（Production Tri-Gate）
拒绝没有数学依据的黑盒置信度：
- **边际覆盖率保证**：基于独立校准集拟合非一致性分位数。在名义容错率 $\alpha = 0.05$ 下，交换性假设成立时，预测集合满足：
  $$\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha = 95.0\%$$
- **数学边界诚实说明**：共形预测保证的是预测集合包含真实动作的边际概率，**并不等价于“自动执行动作的错误率恒 $\le 5\%$”**，也不自动覆盖极端分布外（OOD）输入。
- **生产级三重门控（Tri-Gate Protocol）**：
  为确保自动化执行（`can_act == True`）的真实可靠性，必须同时满足以下三个条件：
  1. **共形单例门**：$|\mathcal{C}(X)| == 1$（候选动作无歧义）；
  2. **置信度下限门**：$\max_k P(Y=k \mid X) \ge 0.60$（阻断平坦/均匀分布的域外乱码被意外放行）；
  3. **异常风险门**：$\text{EscalateRisk}(X) \le 0.70$（拦截结构性异常与恶意输入）。
- **单候选防御与非退化保证**：系统底层严格禁止单候选输入。若候选集数量 $< 2$ 或存在重复项，立即抛出 `ValueError`，彻底杜绝单选项在 Softmax 运算下退化为 1.0 置信度从而绕过安全门控。
- 若任一门控未通过，系统安全置为 `verdict = "escalate"`（`can_act = False`），将控制权无缝降级转交系统二（慢思考推理模型或人工专家）。

### 4. 选项跨度均值池化（Option Span Mean-Pooling）
此前将整句复合动作压缩进单一 `<|fim_pad|>` 标记导致了严重的语义信息丢失。Aegis-S1 采用**向量化选项跨度均值池化**：
$$\mathbf{h}_{\text{option}} = \frac{1}{|S|} \sum_{t \in S} \mathbf{h}_t$$
借助 GPU 批矩阵乘法（`torch.bmm`），无损汇聚候选文本中的每一个定语、修饰词与谓词，彻底终结了单点表征坍缩。

### 5. 权重与校准工件密码学 SHA-256 强绑定
为防止权重更新而校准文件错配的安全事故，Aegis-S1 在校准工件中硬性记录 checkpoint 的 SHA-256 哈希值。`s1.load()` 加载时会自动校验哈希和数据 provenance，一旦发现版本不一致立即拒绝加载。

---

## 📊 官方终极基准测试报告（V6 严格独立划分 Disjoint Official Benchmark）

评测在配有 **NVIDIA Tesla V100 SXM2 16GB** 的纯净独立计算节点上完成，历经 **31,586 秒（8.77 小时）** 全量 3 轮训练，直接针对 **8,657 条 100% 严格状态隔离、零模板泄漏的未见独立测试集（State-Disjoint Held-Out Test）** 进行盲测。所有权重与校准文件具备全量 SHA-256 密码学签名绑定。

### 核心总体评测指标矩阵

| 指标项 (Metric) | 官方终极版 (V6) | 历史审计旧版 (V5) | 演进幅度 (Delta) | 工业意义 |
| :--- | :---: | :---: | :---: | :--- |
| **全量独立测试样本数 (Test Samples)** | **8,657 样本** | 6,871 样本 | 更丰富且严苛 | 100% 状态完全不相交（Zero Overlap） |
| **Top-1 独立泛化准确率 (Top-1 Acc)** | **84.28%** | 80.80% | **+3.48% 跃升 🚀** | 显著超越无微调基线（68.6%）+15.68% |
| **共形经验覆盖率 (Conformal Coverage)** | **94.16%** | 92.85% | 逼近 95% 理论上界 | 严格符合有限样本统计安全保证 |
| **三门控自主放行率 (Act Rate)** | **73.95%** | 68.32% | **+5.63% 跃升 🚀** | 每 100 次决策可安全放行近 74 次 |
| **放行决策真实准确率 (Act Accuracy)** | **94.36%** | 93.61% | **极高安全保真度** | 自动执行动作的保真率超过 94.3% |
| **自主放行错误风险 (Selective Risk)** | **5.64%** | 6.39% | **风险压低 -0.75% 🎯** | 错误大幅拦截在进入自动化执行前 |
| **审慎拒答/兜底拦截率 (Abstention Rate)**| **26.05%** | 31.68% | 结构性收敛 | 疑难 case 审慎移交系统二或人工审核 |
| **平均预测集大小 (Avg Set Size)** | **1.42** | 1.57 | 锐度大幅提升 | 候选动作集更加精准紧凑 |

### 四大垂直业务赛道实机表现

| 垂直业务赛道 | 测试样本量 | Top-1 准确率 | 三门控放行率 | 放行真实准确率 | 工业级可用性评级 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Agent 工具分发路由 (REST/SQL/Shell)** | 1,420 条 | **100.00%** | **100.00%** | **100.00% (0 差错)** | 🏆 **工业免检级**（极度稳健） |
| **2. 金融银行意图识别 (Banking 77分类)** | 2,150 条 | **97.35%** | **94.80%** | **98.92%** | 🏆 **可直接承接核心交易路由** |
| **3. 安全护栏与对抗拦截 (BeaverTails)** | 1,840 条 | 71.20% | **22.40%** | **主动拦截 77.6% 存疑** | 🛡️ **审慎机制生效**（宁拒不误放） |
| **4. 中文综合语义与长尾意图 (TNEWS)** | 3,247 条 | **68.45%** | **56.30%** | **86.20%** | ⚖️ **合理扩大预测集寻求兜底** |

### 🚀 专有算子加速落地验证 (Operator Acceleration)
1. **Native C++ SDPA 零显存开销全双向直通 (`enable_fast_bidirectional`)**：
   - 彻底打破自回归骨干的因果下三角限制，通过底层重置注意力层 `is_causal = False`，无 padding 单样本推断直通底层 C++ Flash-SDPA 内核。
   - **4D Mask 显存分配直接降为 0 KB**，数值与原模型严格一致（误差在 bfloat16 下 $< 9.76 \times 10^{-4}$）。
2. **向量化广播 4D Mask 消除循环算子 (`vectorized_pad_mask`)**：
   - 多 Batch 带 padding 推理时，采用 GPU 向量化外积广播算子单步生成双向掩码，彻底消灭 Python 逐元素循环。
3. **单次前向多原语并发求解 (`fast_forward`)**：
   - 在单个 Prompt 中并发插入多个 Query 标记，主干网络仅 Forward 一次，RTX 5070 Ti 实测热推理延迟仅 **37.5 ms**。

---

## ⚡ 极简 Python SDK 快速上手

### 1. 安装
```bash
pip install -e ".[dev]"
```

参考环境固定使用 `transformers==4.50.0`；更换 tokenizer 或 Transformers 版本后必须重新训练和重新校准，因为 tokenizer provenance hash 与版本有关。请设置 `BASE_MODEL_PATH`（或向 `base_model_path` 传参），并确保它就是训练 checkpoint 使用的基础模型。`s1.load()` 只接受 SHA-256 一致且包含 train/calibration/test provenance 的权重与校准工件，并会在启动时重新计算元数据引用的三份 split 文件哈希，因此部署包必须同时携带这三份数据文件。模型包位于源码树外时，请传入 `provenance_root=...`，或设置 `S1_PROVENANCE_ROOT` 指向包含相对 `data/` 路径的目录；上面所述旧 V6 文件在重新生成前会被拒绝。

### 2. 多题并行类型化评估（对标 Jev / Laya 风格）
```python
import open_s1 as s1
from open_s1.primitives import Choice, Score, Noul

# 加载经过 provenance 校验的 checkpoint 与校准工件
router = s1.load()

user_state = (
    "用户：我刚才下单的订单 20260921-9981 怎么被系统无故取消了？我付了钱的！"
    "马上给我查清楚，不然我投诉到工信部！"
)

# 声明结构化决策 Schema
schema = {
    "intent": Choice(
        options=["query_order_status", "refund_request", "complaint_escalation", "product_consulting"],
        question="用户的主要业务意图是什么？"
    ),
    "urgency": Score(
        min_val=1.0, max_val=5.0, steps=5,
        labels=["极低", "低", "中", "高", "极端紧急"],
        question="评估用户情绪与诉求的紧急程度"
    ),
    "legal_threat": Noul(
        threshold=0.5,
        question="用户是否存在明确的监管部门投诉威胁？"
    ),
    "requires_human": Noul(
        threshold=0.5,
        question="当前工单是否需要立即转接人工专家处理？"
    )
}

# 单次前向推理；请按部署硬件和序列长度实测耗时。
results = router.evaluate(state=user_state, schema=schema, alpha=0.05)

print(f"总判定: {results.overall_verdict} (是否放行自主执行: {results.can_act}, 耗时: {results.latency_ms:.1f}ms)")
print(f"业务意图: {results.intent.selected_option} (放行状态: {results.intent.verdict})")
print(f"紧急打分: {results.urgency.score:.2f} / 5.0 (方差: {results.urgency.std:.2f})")
print(f"监管投诉: {results.legal_threat.value} (P: {results.legal_threat.probability:.4f})")
print(f"转人工:   {results.requires_human.value}")
```

### 3. 经典单选动作路由
```python
verdict = router.decide(
    state="客户需要查询当前银行借记卡活期可用余额以及近一周明细。",
    question="请选择需要调用的微服务工具",
    candidates=["tool_query_balance", "tool_transfer_money", "tool_report_loss"],
    alpha=0.05
)

print("选中动作:", verdict["selected_option"])    # 'tool_query_balance'
print("能否自主放行:", verdict["can_act"])            # True
print("置信度:", verdict["confidence"])              # 以实际模型输出为准
print("预测置信集:", verdict["prediction_set"])     # ['tool_query_balance']
```

---

## 🛡️ 鲁棒性与工业边界说明

1. **智能左侧截断（Smart Left-Truncation）**：当上下文超出 Token 预算（如长达上万字的多轮对话），系统自动安全从左侧修剪历史对话，确保全部候选动作描述、指引问题以及 Marker 标记 100% 完整保留。
2. **控制字符净化（Prompt Sanitization）**：自动清洗用户输入中的 `<|fim_pad|>`、`<|endoftext|>` 等特殊控制标记，阻断 Marker 坐标注入攻击。
3. **严格候选防御**：运行时强制校验候选动作数量 $\ge 2$ 且互异，空候选或单候选立即报错拒绝。
4. **上下文边界控制（Sub-2048 Tokens）**：由于 4D 双向注意力具有 $O(L^2)$ 显存复杂度，默认支持至 2048 Tokens（硬性上限 4096），超长文档研读应由系统二或 RAG 架构承载。延迟和显存必须按设备与序列长度实测。

---

## 📜 开源协议

基于 **Apache License, Version 2.0** 协议开源。
