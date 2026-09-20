# -*- coding: utf-8 -*-
"""
广范围多领域中文决策能力评测与原始结果导出工具
覆盖四大业务场景（纯安全、无违规词汇）：
1. 办公协同与工具调度（Excel数据、飞书/企微通知、日程、搜索、代码绘图）
2. 多轮对话完整性与停嘴判定（思考停顿、未完待续、闭合指令）
3. 客服意图分类与工单路由（售后、售前、账户、人工服务）
4. 模糊边界与拒识防爆（输入条件不全，考察模型是否老实报警而不是瞎猜）
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from open_s1 import AegisRouter

OUTPUT_FILE = Path("E:/s1-decision-model/data/audit_results_zh.json")

# 详尽的评测题库（正例、反例、边界模糊题兼备）
TEST_SUITE: List[Dict] = [
    # ---------------- 场景一：办公协同与工具调用 ----------------
    {
        "id": "office_01",
        "category": "办公工具路由",
        "state": "用户说：请把上个季度各大区销售额汇总表以 Excel 形式生成，并标注出同比下滑的部门。",
        "question": "系统接下来应该调度哪项核心能力？",
        "options": [
            "excel_generator: 数据聚合计算并格式化生成结构化电子表格",
            "web_search: 在公共互联网上检索公开资讯",
            "calendar_event: 创建团队日程会议与提醒",
            "voice_recording: 开启本地麦克风录音"
        ],
        "target_idx": 0,
        "target_label": "excel_generator"
    },
    {
        "id": "office_02",
        "category": "办公工具路由",
        "state": "用户说：帮我在明天下班前给项目攻坚组全员发送一条飞书加急通知，提醒大家提交周报。",
        "question": "应该调用哪一个协作工具？",
        "options": [
            "calculator: 执行四则数学公式运算",
            "im_bot_notify: 通过企业即时通讯（飞书/企微）机器人发送群通知",
            "excel_generator: 生成电子表格文件",
            "pdf_watermark: 给本地文档添加防泄密水印"
        ],
        "target_idx": 1,
        "target_label": "im_bot_notify"
    },
    {
        "id": "office_03",
        "category": "办公工具路由",
        "state": "用户说：根据这两组实验数据，用 Python 的 seaborn 库绘制一个散点回归拟合图。",
        "question": "应该调用哪个工具？",
        "options": [
            "code_interpreter: 在沙盒环境中执行代码并生成可视化图表",
            "im_bot_notify: 发送企业群通知",
            "calendar_event: 创建会议日程",
            "voice_recording: 开启录音"
        ],
        "target_idx": 0,
        "target_label": "code_interpreter"
    },
    {
        "id": "office_04",
        "category": "办公工具路由",
        "state": "用户说：明天下午三点约人事部张经理在第三会议室进行绩效沟通，帮我占下会议室。",
        "question": "应该选择哪个工具？",
        "options": [
            "web_search: 搜索互联网网页",
            "code_interpreter: 运行数据分析脚本",
            "calendar_event: 预订会议室并创建日历日程事件",
            "excel_generator: 制作销售电子表格"
        ],
        "target_idx": 2,
        "target_label": "calendar_event"
    },
    {
        "id": "office_05",
        "category": "办公工具路由",
        "state": "用户说：查一下我们公司知识库里关于‘出差餐补与住宿报销标准’的内部制度规定。",
        "question": "应该调用哪个模块？",
        "options": [
            "enterprise_rag_search: 检索企业私有向量知识库与制度文档",
            "calculator: 计算数学数值公式",
            "im_bot_notify: 发送工作群通知",
            "excel_generator: 生成表格"
        ],
        "target_idx": 0,
        "target_label": "enterprise_rag_search"
    },

    # ---------------- 场景二：对话完整性与端点停嘴判断 ----------------
    {
        "id": "dialogue_01",
        "category": "多轮对话完整性",
        "state": "[助手]: 请问您需要查询哪天的门票？\n[用户]: 嗯……我想想啊，大概是下周三……或者是……",
        "question": "结合语境，用户是否已经表达完毕？",
        "options": [
            "incomplete: 处于犹豫思考或话语未完状态，系统应保持静默等待",
            "complete: 语义表达清晰闭合，系统应立即切断麦克风响应",
            "clarification: 语义存在矛盾，需要主动反问打断"
        ],
        "target_idx": 0,
        "target_label": "incomplete"
    },
    {
        "id": "dialogue_02",
        "category": "多轮对话完整性",
        "state": "[助手]: 已为您勾选两张成人票，请问还需要加购景区观光车吗？\n[用户]: 不需要了，直接提交订单吧。",
        "question": "结合语境，用户是否已经表达完毕？",
        "options": [
            "complete: 语义表达清晰闭合，系统应立即切断麦克风响应",
            "incomplete: 处于犹豫思考或话语未完状态，系统应保持静默等待",
            "clarification: 语义存在矛盾，需要主动反问打断"
        ],
        "target_idx": 0,
        "target_label": "complete"
    },
    {
        "id": "dialogue_03",
        "category": "多轮对话完整性",
        "state": "[助手]: 请告诉我您的收件人姓名和手机号码。\n[用户]: 我叫李雷。",
        "question": "针对多槽位提问，用户回答了部分信息，系统应如何判断？",
        "options": [
            "incomplete: 多槽位信息未给全，用户可能在看手机准备补充，系统应短暂等待",
            "complete: 用户已说完，直接按现有信息结算",
            "cancel_task: 判定为用户拒绝配合，取消订单"
        ],
        "target_idx": 0,
        "target_label": "incomplete"
    },
    {
        "id": "dialogue_04",
        "category": "多轮对话完整性",
        "state": "[助手]: 请问这个方案满意吗？\n[用户]: 行，可以。",
        "question": "用户简短回答，系统应如何决断？",
        "options": [
            "complete: 简短肯定答复，语义完整闭合，立即进入下一步",
            "incomplete: 字数太少，判定为未说完继续等待",
            "clarification: 语义模糊需重新询问"
        ],
        "target_idx": 0,
        "target_label": "complete"
    },

    # ---------------- 场景三：客服意图分流与工单分类 ----------------
    {
        "id": "intent_01",
        "category": "客服意图分流",
        "state": "用户咨询：买的这件衣服尺码买小了，穿上太紧，能不能帮我换一件大一号的 L 码？",
        "question": "该工单应归属于哪种售后分类？",
        "options": [
            "exchange_goods: 尺码不合或质量换货申请",
            "shipping_delay: 物流停滞与丢件催单",
            "account_security: 密码找回与账号安全",
            "product_inquiry: 售前产品参数咨询"
        ],
        "target_idx": 0,
        "target_label": "exchange_goods"
    },
    {
        "id": "intent_02",
        "category": "客服意图分流",
        "state": "用户留言：物流信息显示三天前就已经在派送中了，到现在都没收到，电话也打不通！",
        "question": "该工单应归属于哪种分类？",
        "options": [
            "shipping_delay: 物流停滞与催促派送",
            "exchange_goods: 尺码换货",
            "invoice_request: 开具增值税发票",
            "account_security: 账号安全"
        ],
        "target_idx": 0,
        "target_label": "shipping_delay"
    },
    {
        "id": "intent_03",
        "category": "客服意图分流",
        "state": "用户在智能客服聊天框输入：我要人工！别给我转机器人，马上接人工坐席！",
        "question": "系统应立即触发哪种路由动作？",
        "options": [
            "transfer_human_agent: 立即流转至人工客服排队队列",
            "push_faq_articles: 强行推送常见问题帮助文档",
            "close_session: 直接结束会话并提示好评",
            "restart_bot: 重启对话机器人流程"
        ],
        "target_idx": 0,
        "target_label": "transfer_human_agent"
    },
    {
        "id": "intent_04",
        "category": "客服意图分流",
        "state": "用户提问：我原本绑定的旧手机号已经注销了，现在接收不到短信验证码，无法登录怎么办？",
        "question": "该工单的分类应该是？",
        "options": [
            "account_security: 账号换绑与实名身份验证找回",
            "exchange_goods: 换货申请",
            "shipping_delay: 物流催单",
            "invoice_request: 发票申请"
        ],
        "target_idx": 0,
        "target_label": "account_security"
    },

    # ---------------- 场景四：模糊边缘与自知之明防爆测试 ----------------
    {
        "id": "edge_01",
        "category": "边界防爆与拒识",
        "state": "用户仅输入了一个字：‘好。’（上下文没有任何既往对话背景）",
        "question": "在缺乏上下文的情况下，判断用户意图：",
        "options": [
            "confirm_action: 确认执行上文操作",
            "greeting_hello: 日常问候",
            "ambiguous_need_clarify: 孤立单字无法判定真实意图，需提示用户补充具体需求",
            "cancel_action: 取消操作"
        ],
        "target_idx": 2,
        "target_label": "ambiguous_need_clarify"
    },
    {
        "id": "edge_02",
        "category": "边界防爆与拒识",
        "state": "用户输入：‘帮我把那个弄一下。’（未指定对象、未指定动作、无指代目标）",
        "question": "系统应该做出什么响应判断？",
        "options": [
            "ambiguous_need_clarify: 指代严重缺失，无法自动化决策，必须请求用户澄清",
            "delete_all_files: 清理全部临时文件",
            "reboot_system: 重启系统",
            "submit_order: 提交订单"
        ],
        "target_idx": 0,
        "target_label": "ambiguous_need_clarify"
    },
    {
        "id": "edge_03",
        "category": "边界防爆与拒识",
        "state": "用户输入：‘用 C++ 重构 Unreal Engine 5 的 Lumen 光照缓存着色器。’",
        "question": "在日常办公助手环境中，评估该任务的可执行性：",
        "options": [
            "beyond_capability_escalate: 超出当前轻量助手能力范围，需转交专业图形工程系统",
            "simple_chat: 属于闲聊日常话题",
            "calculator: 使用计算器处理",
            "weather_api: 查询天气"
        ],
        "target_idx": 0,
        "target_label": "beyond_capability_escalate"
    }
]


def run_test():
    print("[*] 正在加载本地 Aegis-S1 模型...")
    router = AegisRouter.load()
    print("[*] 模型加载完成，开始执行全领域原始测试...\n")

    results = []
    correct_count = 0
    total = len(TEST_SUITE)

    for s in TEST_SUITE:
        t0 = time.perf_counter()
        out = router.decide(
            state=s["state"],
            question=s["question"],
            candidates=s["options"]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred_idx = out["selected_index"]
        target_idx = s["target_idx"]
        is_correct = (pred_idx == target_idx)
        if is_correct:
            correct_count += 1

        record = {
            "id": s["id"],
            "category": s["category"],
            "state": s["state"],
            "question": s["question"],
            "options": s["options"],
            "target_label": s["target_label"],
            "selected_option": out["selected_option"],
            "selected_index": pred_idx,
            "target_index": target_idx,
            "is_correct": is_correct,
            "confidence": out["confidence"],
            "conformal_verdict": out["conformal_verdict"],  # 'act', 'escalate', 'reject'
            "prediction_set": out["prediction_set"],
            "latency_ms": round(elapsed_ms, 2),
            "probabilities": out["probabilities"],
        }
        results.append(record)

    # 统计指标
    acc = correct_count / total
    print(f"[*] 综合评测完毕！测试题目数: {total} | 准确率: {acc:.1%}")

    # 保存原始中文 JSON 评测记录
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "correct_count": correct_count,
            "accuracy": round(acc, 4),
            "records": results
        }, f, ensure_ascii=False, indent=2)

    print(f"[*] 完整原始结果已保存在: {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run_test()
