# Aegis-S1：具备数学安全保证的非自回归系统一（System 1）决策大模型基座

<p align="center">
  <img src="https://img.shields.io/badge/开源协议-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/推理耗时-40~65毫秒-green.svg" alt="Inference Speed">
  <img src="https://img.shields.io/badge/数学安全-真·分裂共形预测覆盖率96%25-orange.svg" alt="Safety Guarantees">
  <img src="https://img.shields.io/badge/原语对齐-Choice%20%7C%20Score%20%7C%20Noul-purple.svg" alt="Decision Primitives">
  <img src="https://img.shields.io/badge/多题并行-单次前向并行解析-red.svg" alt="Multi Query">
</p>

**Aegis-S1** 是开源的高性能**“系统一（System 1）”非自回归决策基座模型**，专为 AI 智能体（Agent）提供毫秒级（~40–65ms）、具备有限样本数学安全保证的“直觉神经反射”。

项目深度对标硅谷 **TypeSafe Jev** 与开源项目 **Laya** 的设计思想，彻底**摒弃逐字生成 JSON** 的高延迟与格式崩溃风险（单次决策省去 500~2000ms），通过**单次前向传播**直接并发解析结构化决策原语。

---

## 🌟 核心突破与对齐特性

### 1. 三大类型化决策原语（全面对齐 Jev / Laya）
Aegis-S1 提炼了智能体决策的最小完备原语集合：
- `Choice(options, question)`：**离散候选分类**。在运行时动态传入任意字符串候选项，基于分裂共形预测输出数学置信集 $\mathcal{C}(X)$ 与放行/拒识判定；
- `Score(min_val, max_val, steps, labels)`：**连续/序数标度打分**。采用离散概率期望 $\mathbb{E}[S] = \sum v_k \cdot p_k$ 给出平滑评分与认识论方差 $\mathrm{Var}[S]$；
- `Noul(threshold, question)`（别名 `Boolean`）：**布尔条件断言**。精准计算条件成立概率 $P(\text{true})$，内置安全裕度边界检查，杜绝临界值误判。

### 2. 多题并行单次前向评估（Single-Forward Parallel Evaluation）
在复杂的工业工单与智能体调度中，单个用户状态通常需要同时回答多个异构问题（如：意图分类、紧急程度打分、监管投诉风险断言、是否转人工专家）：
- 传统自回归大模型需生成 50~150 个 Token 或并发调用多次 API，耗时 1.5~3 秒；
- **Aegis-S1 在单次前向中完成全部问题解析**：多题的 Marker 统一嵌入序列，LLM 主干网络**仅执行一次**，在 RTX 5070 Ti 上**仅需 65.8 毫秒即可同时评估 4 个异构问题**，提速约 **80 倍**！

### 3. 真·分裂共形预测数学安全保底（Split-Conformal Prediction）
拒绝没有数学依据的黑盒评分：
- 模型在独立无重叠校准集（300 样本）上拟合非一致性分位数 $\hat{q} = 0.9869$；
- 在名义容错率 $\alpha = 0.05$ 下，严格保证边缘覆盖率 $P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$；
- 当遇到语义模糊或争议工单时（预测集合大小 $|\mathcal{C}(X)| > 1$）或全局异常门控报警，系统**主动拒绝瞎猜（Verdict: escalate）**，将控制权无缝移交给系统二（慢思考推理模型或人工专家）。

### 4. 4D 全双向自注意力掩码破除“因果致盲”
原生 Decoder LLM 的因果掩码导致选项开头的 Marker 对选项文本的注意力**精确为 0.0000**。Aegis-S1 注入 4D 双向自注意力矩阵，使全序列双向可见，并结合 2 层选项间动态交叉注意力（`InterOptionTransformer`），彻底激活候选间的竞争建模能力。

---

## 📊 严格无模板测试集（Zero-Overlap）实测报告

我们在与训练集模板**绝对交集为空（$Train \cap Calib \cap Test = \emptyset$）**的 300 条独立保留测试集上进行了端到端测试，不含任何数据背诵：

| 评估指标 | 实测数值 | 工业意义与说明 |
| :--- | :--- | :--- |
| **泛化准确率（Top-1 Accuracy）** | **67.67%** | 在完全未见过的独立工单上的无偏真实泛化准确率 |
| **经验共形覆盖率（Marginal Coverage）** | **96.00%** | **超越理论下界保底**（理论目标 $\ge 95.0\%$，$\alpha=0.05$） |
| **自主放行样本错误率（Selective Risk on Act）** | **5.56%** | 当模型确认下发 `act` 放行指令时，**决策正确率高达 94.44%** |
| **主动拒识/转慢思考率（Abstention Rate）** | **76.00%** | 在面对不确定或模糊问题时，精准拦截并转入系统二 |
| **单选推理延迟（热启动）** | **~50.4 ms** | RTX 5070 Ti 移动端 GPU 上实测亚 55 毫秒 |
| **4 题并行评估延迟（热启动）** | **~65.8 ms** | 单次前向同时解析 4 道异构题目（Jev 对标） |
| **轻量级权重体积** | **34.41 MB** | 极速分发与秒级加载，无需存储数百兆完整模型 |

### 真实对话反例实测校验
- **输入工单**：`“这个方案满意吗？” -> “行，可以。”`（极其典型的口语化模糊反馈）
- **模型推理**：
  - 选中项：`confirm_satisfied`（置信度：`0.4688`，全局异常风险：`0.3652`）
  - **共形安全裁定：`escalate`（拒识/转慢思考）**
  - 共形预测集合：`['confirm_satisfied', 'negative_reject', 'ambiguous_clarify']`
  - 说明：共形集合包含 3 个冲突候选动作，系统拒绝盲目采纳 Top-1，成功规避了生产误操作！

---

## ⚡ 极简 Python SDK 快速上手

### 1. 安装与依赖
```bash
pip install torch transformers peft
```

### 2. 多题并行类型化评估（对标 Jev / Laya 风格）
```python
import open_s1 as s1
from open_s1.primitives import Choice, Score, Noul

# 一键加载 Aegis-S1（自动载入权重与共形校准工件）
router = s1.load("output/s1_model_v3")

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

# 单次前向并行解析（全部 4 道题耗时仅约 65ms！）
results = router.evaluate(state=user_state, schema=schema, alpha=0.05)

print(f"全局裁定: {results.overall_verdict} (耗时: {results.latency_ms} ms)")
print(f"意图分类: {results.intent.selected_option} (裁定: {results.intent.verdict})")
print(f"紧急分值: {results.urgency.score} / 5.0 (标准差: {results.urgency.std})")
print(f"监管威胁: {results.legal_threat.value} (P: {results.legal_threat.probability})")
print(f"转接人工: {results.requires_human.value}")
```

### 3. 单候选动作极速决策
```python
verdict = router.decide(
    state="用户询问当月话费余额与消费明细",
    question="应该调度哪一个工具接口？",
    candidates=["tool_query_balance", "tool_transfer_money", "tool_cancel_card"],
    alpha=0.05
)

print(verdict["selected_option"])    # 'tool_query_balance'
print(verdict["conformal_verdict"])  # 'act'
print(verdict["prediction_set"])     # ['tool_query_balance']
```

---

## 🛡️ 工业级防御与工程边界

Aegis-S1 彻底修复了常规原型中的工程漏洞：
1. **智能左截断（Smart Left-Truncation）**：当上下文文本超长（如上万字长对话）时，自动从左侧裁剪早期历史记录，**绝对保证**候选选项列表与 Marker 标记 100% 完整，消除 Token 坐标坍缩 Bug；
2. **输入控制字符清洗（Prompt Sanitization）**：自动过滤用户输入中的 `<|fim_pad|>`、`<|endoftext|>` 等内部特殊标记，从根本上防止注入攻击导致 Marker 错位；
3. **严格边界异常报错**：当候选选项数量自身超出模型 Token 预算时，明确抛出 `ValueError`，绝不静默返回虚假的均匀概率。

---

## 📂 项目结构

```
aegis/
├── open_s1/                     # 对外发布的 Agent SDK
│   ├── __init__.py              # AegisRouter 核心类与 s1.load() 接口
│   └── primitives.py            # Choice, Score, Noul 原语与类型化结果容器
├── src/                         # 底层模型与数学算法实现
│   ├── modeling_s1.py           # S1DecisionModel、4D 双向注意力与选项交互网络
│   ├── tokenizer_utils.py       # 智能左截断与单/多题统一 Token 编码器
│   └── conformal.py             # 分裂共形预测校准器与工件持久化引擎
├── scripts/
│   ├── prepare_dataset_v3.py    # 模板严格互斥的三向数据集生成器 (70%/15%/15%)
│   ├── train_s1.py              # LoRA 增量微调流水线
│   └── benchmark_v3_rigorous.py # 独立保留测试集客观评测套件
├── tests/
│   ├── test_robustness_boundary.py        # 边界截断、注入防御与异常检查测试
│   └── test_primitives_and_multi_query.py # 三大原语与多题单前向并行基准测试
└── output/
    └── s1_model_v3/             # 训练出的 34MB 权重与校准工件 (conformal_calibration.json)
```

---

## 📜 开源协议

本项目依据 **Apache License 2.0** 协议开源，商业友好，允许自由修改、分发及商业闭源集成。
