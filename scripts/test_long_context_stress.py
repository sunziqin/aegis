# -*- coding: utf-8 -*-
"""
审稿人级硬核压力测试：
1. 真实部署的开源版 Jev (Laya) 对比实测
2. 4000~6000字超长上下文“大海捞针”（Needle-in-a-Haystack）注意力缺陷检验
   - 测试点：关键决策指令埋藏在上下文第 10% (前部)、50% (中部-测试Lost in the Middle)、90% (后部)
   - 对比 Laya (512字截断) vs Aegis-S1 (长上下文注意力) 的真实表现与注意力衰减！
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_s1 import AegisRouter
from laya import Router as LayaRouter

OUTPUT_LOG = Path("E:/s1-decision-model/data/long_context_stress_results.json")

# 生成真实干扰长文本（模拟长达 5000+ 字的复杂系统技术规范、日志与规则文档）
HAYSTACK_PARAGRAPHS = [
    "第 {idx} 节：分布式对象存储系统的底层数据同步策略与读写一致性机制。在多可用区跨机房同步场景下，系统默认采用 Raft 共识算法进行 WAL 日志回放。每个 StorageNode 会周期性向 Coordinator 发送心跳检测报文，若心跳丢失超过 15 秒则标记为降级状态。",
    "第 {idx} 节：网关层流量染色与灰度路由调度协议。当收到外部 HTTP 请求时，接入层 Envoy 代理会根据 Header 头部中的 x-canary 标签进行流量镜像与加权分流。对于未携带标签的常规请求，流量均分转发至默认集群池。",
    "第 {idx} 节：关系型数据库连接池在高并发场景下的超时重试与背压机制。数据库连接超时时间建议设定为 3000 毫秒，当连接池耗尽时，应用端应捕获 ConnectionPoolExhaustedException 并执行指数退避重试，禁止无限递归重连。",
    "第 {idx} 节：本地缓存与分布式 Redis 缓存的双写一致性保障策略。对于热点读商品信息，优先从本地 Caffeine 缓存读取，未命中则穿透至 Redis 集群。当执行写操作时，采用先更新数据库再延时双删 Redis 的标准流程。"
]


def generate_long_document(needle_text: str, depth_ratio: float, total_paragraphs: int = 40) -> str:
    """
    在长文本中指定深度（depth_ratio: 0.1, 0.5, 0.9）埋入关键核心决策规则（Needle）
    """
    paragraphs = []
    needle_pos = int(total_paragraphs * depth_ratio)
    
    for i in range(total_paragraphs):
        if i == needle_pos:
            paragraphs.append(f"\n【★★★ 特别紧急最高优先级安全指令 ★★★】\n{needle_text}\n")
        else:
            p_template = HAYSTACK_PARAGRAPHS[i % len(HAYSTACK_PARAGRAPHS)]
            paragraphs.append(p_template.format(idx=i+1))
            
    return "\n\n".join(paragraphs)


def run_stress_test():
    print("=" * 75)
    print("  审稿人级残酷压力测试：Laya (开源Jev) vs Aegis-S1 长文本注意力深度审查  ")
    print("=" * 75)

    # 1. 真实部署开源 Jev (Laya)
    print("[*] 正在加载本地已部署的开源版 Jev (Laya Router)...")
    laya_router = LayaRouter()
    print("[*] Laya 加载成功！")

    # 2. 加载 Aegis-S1
    print("[*] 正在加载本地 Aegis-S1 (Qwen 架构)...")
    s1_router = AegisRouter.load()
    print("[*] Aegis-S1 加载成功！")

    # 3. 构造长文本大海捞针样本
    # 核心针：埋在数千字文档中
    needle = "系统特别审计红线第 99 条：一旦检测到告警代码为 [ERR-ALPHA-99]，严禁执行任何重启动作，必须立即调用 isolate_network_switch 执行核心交换机物理断网！"
    
    test_depths = [
        ("前部 (10% 深度)", 0.1),
        ("中部 (50% 深度 - 审稿人重点测试 Lost-in-the-Middle 缺陷)", 0.5),
        ("后部 (90% 深度)", 0.9)
    ]

    options = [
        "restart_cluster_nodes: 重启整个分布式计算节点集群",
        "isolate_network_switch: 紧急切断核心交换机物理网络连接",
        "ignore_and_continue: 告警级别低，忽略并保持运行",
        "export_error_log: 将错误日志导出至本地磁盘"
    ]
    target_idx = 1
    target_label = "isolate_network_switch"

    results = []

    for depth_name, depth_val in test_depths:
        print("\n" + "-" * 75)
        print(f"[*] 正在生成【{depth_name}】长文本压测样本...")
        long_doc = generate_long_document(needle, depth_val, total_paragraphs=35)
        doc_char_len = len(long_doc)
        print(f"[*] 生成长文本长度: {doc_char_len} 字符 (约 2500~3000 Token)")

        query_state = f"【系统运行文档与操作守则】\n{long_doc}\n\n【突发实时事件】：监控中心上报严重异常，错误码为 [ERR-ALPHA-99]，工程师请求下一步动作指导。"
        query_question = "结合上方守则的特别紧急指令，系统必须采取哪项动作？"

        # --- A. 测试开源 Jev (Laya) ---
        print(f"\n[1] 正在用 开源版 Jev (Laya) 运行测试...")
        laya_start = time.perf_counter()
        laya_error = None
        laya_choice = None
        laya_conf = None
        
        try:
            # 格式化为 Laya 的标准输入
            laya_state = {"document": query_state}
            laya_questions = {
                "decision": {
                    "type": "choice",
                    "instructions": query_question,
                    "criteria": {
                        "restart": "重启节点",
                        "isolate": "切断交换机网络",
                        "ignore": "忽略告警",
                        "export": "导出日志"
                    }
                }
            }
            laya_res = laya_router.predict(laya_state, laya_questions)
            laya_ans = laya_res["answers"]["decision"]
            laya_choice = laya_ans.get("choice")
            laya_conf = laya_ans.get("confidence")
            laya_time_ms = (time.perf_counter() - laya_start) * 1000
            print(f"  -> Laya 输出选择: {laya_choice} (把握度: {laya_conf}) | 耗时: {laya_time_ms:.1f}ms")
            print(f"  -> Laya 内部截断信息: 输入Token数={laya_res.get('usage', {}).get('input_tokens')}")
        except Exception as e:
            laya_error = str(e)
            laya_time_ms = (time.perf_counter() - laya_start) * 1000
            print(f"  -> Laya 运行崩溃或报错: {e}")

        # --- B. 测试 Aegis-S1 ---
        print(f"\n[2] 正在用 Aegis-S1 (长文本) 运行测试...")
        s1_start = time.perf_counter()
        s1_out = s1_router.decide(
            state=query_state,
            question=query_question,
            candidates=options,
            max_length=4096,  # 开启大长文本视野
        )
        s1_time_ms = (time.perf_counter() - s1_start) * 1000
        
        s1_pred_idx = s1_out["selected_index"]
        s1_is_correct = (s1_pred_idx == target_idx)
        print(f"  -> Aegis-S1 输出选择: {s1_out['selected_option']}")
        print(f"  -> Aegis-S1 是否命中关键针: {'【命中正确】' if s1_is_correct else '【致命丢失/判断错误】'}")
        print(f"  -> Aegis-S1 把握度: {s1_out['confidence']:.4f} | 拒识风险: {s1_out['escalate_risk']:.4f}")
        print(f"  -> Aegis-S1 共形裁定: {s1_out['conformal_verdict']} | 耗时: {s1_time_ms:.1f}ms")
        print(f"  -> 各选项概率分布: {s1_out['probabilities']}")

        results.append({
            "depth": depth_name,
            "char_length": doc_char_len,
            "target": target_label,
            "laya_result": {
                "choice": laya_choice,
                "confidence": laya_conf,
                "error": laya_error,
                "time_ms": round(laya_time_ms, 2)
            },
            "s1_result": {
                "selected": s1_out["selected_option"],
                "selected_index": s1_pred_idx,
                "is_correct": s1_is_correct,
                "confidence": s1_out["confidence"],
                "escalate_risk": s1_out["escalate_risk"],
                "verdict": s1_out["conformal_verdict"],
                "time_ms": round(s1_time_ms, 2),
                "probs": s1_out["probabilities"]
            }
        })

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 75)
    print(f"[*] 残酷压力测试完成！原始 JSON 数据记录在: {OUTPUT_LOG}")
    print("=" * 75)


if __name__ == "__main__":
    run_stress_test()
