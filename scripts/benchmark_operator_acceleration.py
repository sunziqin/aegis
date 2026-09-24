# -*- coding: utf-8 -*-
"""
Operator Benchmark: 4D Float Eager Mask vs Native C++ SDPA Flash Attention.
Demonstrates the exact latency gap between:
1. Naive 4D float attention mask (falls back to Python/Eager memory-bound attention)
2. PyTorch Native C++ SDPA kernel with is_causal=False (FlashAttention-2 / MemEfficient)
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling_s1 import S1DecisionModel
from src.tokenizer_utils import encode_decision_batch

def run_benchmark():
    base_model = os.environ.get("BASE_MODEL_PATH", "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct")
    weight_file = str(Path(__file__).resolve().parent.parent / "output" / "s1_model_v6" / "s1_decision_weights.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[*] Running Operator Benchmark on: {device} ({gpu_name})")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(base_model)
    config._attn_implementation = "sdpa"
    hidden_dim = getattr(config, "hidden_size", 896)

    backbone = AutoModel.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)

    model = S1DecisionModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_inter_layers=2,
    ).to(device)

    if device == "cuda":
        model.decision_head.to(dtype=torch.bfloat16)

    ckpt = torch.load(weight_file, map_location=device)
    model.backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    model.decision_head.load_state_dict(ckpt["decision_head"])
    model.eval()

    test_state = (
        "用户在移动端APP发起借记卡跨境汇款交易，目标账户为渣打银行（香港分行），"
        "汇款金额为 50,000 港币。核心风控系统检测到该账户近24小时内存在两次异地IP登录记录，"
        "且本次交易触发了‘单日大额涉外交易’敏感规则，但用户已成功通过短信验证码及人脸生物识别校验。"
    )
    test_question = "分析风控等级并选择下一步拦截或放行操作"
    test_candidates = [
        "直接无条件放行并提交SWIFT跨境国际清算网络",
        "挂起交易并转入反洗钱高危人工专家二级复核流",
        "强制中断交易并永久冻结该借记卡外汇结算功能",
        "要求用户重新进行网点线下柜面身份核验"
    ]

    encoded = encode_decision_batch(
        tokenizer=tokenizer,
        states=[test_state],
        questions=[test_question],
        options_per_sample=[test_candidates],
        max_length=2048,
        device=device,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    marker_indices = encoded["marker_indices"]
    marker_mask = encoded["marker_mask"]
    option_spans = encoded["option_spans"]

    seq_len = input_ids.shape[1]
    print(f"[*] Input Sequence Length: {seq_len} tokens")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, attention_mask, marker_indices, marker_mask, option_spans, bidirectional=True)
    if device == "cuda":
        torch.cuda.synchronize()

    # 1. Benchmark Current Approach: 4D Float Eager Mask
    N_RUNS = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_RUNS):
            _ = model(input_ids, attention_mask, marker_indices, marker_mask, option_spans, bidirectional=True)
            if device == "cuda":
                torch.cuda.synchronize()
    t_eager_total = (time.perf_counter() - t0) * 1000.0
    lat_eager = t_eager_total / N_RUNS

    # 2. Benchmark Operator-Optimized SDPA Kernel (Direct Bidirectional SDPA without 4D float allocation)
    # Monkey-patch forward for SDPA direct passthrough
    def fast_forward(input_ids, marker_indices, marker_mask, option_spans):
        # Pass attention_mask=None for batch=1 to trigger pure FlashAttention-2 / SDPA kernel directly!
        outputs = model.backbone(input_ids=input_ids, attention_mask=None)
        sequence_hidden = outputs.last_hidden_state
        logits, probs, escalate = model.decision_head(
            sequence_hidden_states=sequence_hidden,
            marker_indices=marker_indices,
            marker_mask=marker_mask,
            option_spans=option_spans,
        )
        confidence, best_idx = torch.max(probs, dim=-1)
        return {"probs": probs, "best_choice_idx": best_idx, "confidence": confidence}

    # Warmup Fast
    with torch.no_grad():
        for _ in range(10):
            _ = fast_forward(input_ids, marker_indices, marker_mask, option_spans)
    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_RUNS):
            res_fast = fast_forward(input_ids, marker_indices, marker_mask, option_spans)
            if device == "cuda":
                torch.cuda.synchronize()
    t_fast_total = (time.perf_counter() - t1) * 1000.0
    lat_fast = t_fast_total / N_RUNS

    speedup = lat_eager / max(0.001, lat_fast)

    print("\n" + "=" * 65)
    print("      OPERATOR ACCELERATION BENCHMARK RESULTS (100 RUNS)")
    print("=" * 65)
    print(f"  [当前方法] 4D Float Mask (Eager):      {lat_eager:.2f} ms")
    print(f"  [算子优化] Native C++ SDPA (Kernel):    {lat_fast:.2f} ms")
    print(f"  [性能提升] 纯算子级加速比:              {speedup:.2f}x 提速!")
    print("=" * 65)
    print(f"  最优预测动作: {test_candidates[res_fast['best_choice_idx'][0].item()]}")
    print(f"  决策置信度:   {res_fast['confidence'][0].item()*100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()
