# -*- coding: utf-8 -*-
"""
Dataset Preparation V3: 100% Template-Disjoint & Multi-Domain Partition.
Strict separation:
- Train Templates: used ONLY in train_v3.json
- Val/Calib Templates: used ONLY in calib_v3.json
- Test Templates: used ONLY in test_v3.json (includes Banking77-style natural queries and dialogue feedback)
Train ∩ Calib ∩ Test = ∅ (ZERO template text overlap!)
"""

import json
import random
from pathlib import Path
from typing import List, Dict

OUTPUT_DIR = Path("E:/s1-decision-model/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(2026)

# ==============================================================================
# Domain 1: Triage Categories
# ==============================================================================
TRIAGE_CATALOG = [
    ("billing_duplicate", "duplicate charge or overbilled invoice needing reversal / 重复扣款或账单多收需要退还"),
    ("billing_refund", "refund request for accidental purchase, dissatisfaction, or return / 意外购买或商品退款申请"),
    ("subscription_cancel", "customer wants to completely cancel or terminate subscription / 客户明确要求解约注销服务"),
    ("subscription_downgrade", "customer wants to switch to cheaper or lower tier plan / 客户希望降级为低价或免费套餐"),
    ("tech_bug", "software glitch, UI rendering error or unexpected 500 error / 软件界面故障、前端异常或系统报错"),
    ("tech_outage", "entire service, API or website is completely down or unreachable / 核心服务或API整体宕机瘫痪"),
    ("security_compromise", "suspected hacked account, unauthorized login or credential leak / 账户被盗、异常异地登录或安全风险"),
    ("sales_inquiry", "enterprise pricing, volume licensing, quotes or contract negotiation / 企业采购咨询、报价与大客户定制"),
    ("gdpr_compliance", "personal data deletion, export or privacy compliance request / 个人隐私数据彻底删除或导出合规申请"),
    ("account_login_issue", "forgot password, 2FA code not received or login lockout / 忘记密码、双重验证验证码未收到或锁号"),
]

# DISJOINT TEMPLATES FOR TRAIN
TRAIN_TRIAGE_TEMPLATES = [
    ("I noticed a double payment of ${money} on transaction #{num}. Please refund the extra charge.", "billing_duplicate"),
    ("查账发现上个月扣了两次账单，单号 #{num}，请原路返还重复扣费。", "billing_duplicate"),
    ("I accidentally subscribed to the yearly plan. Please cancel and issue a full refund.", "billing_refund"),
    ("七天无理由退款，刚买的会员完全不符合预期，申请全额退款。", "billing_refund"),
    ("I am quitting your platform. Delete my subscription and never charge my card again.", "subscription_cancel"),
    ("请彻底注销我的会员服务并关闭自动扣费，后续不再使用。", "subscription_cancel"),
    ("Our team budget got cut. Can we downgrade from Enterprise to Starter tier?", "subscription_downgrade"),
    ("我们需要把套餐从旗舰版降级为基础个人版，请问怎么操作？", "subscription_downgrade"),
    ("The mobile application throws a null pointer exception when rendering JPEG images.", "tech_bug"),
    ("前端表格组件在按日期排序时发生页面白屏报错，控制台提示 TypeError。", "tech_bug"),
    ("SEV-1 ALERT: All production gateway endpoints are returning 504 Gateway Timeout!", "tech_outage"),
    ("全国机房集群出现大面积网络中断，所有微服务节点健康检查均不通过！", "tech_outage"),
    ("An unknown login occurred from an unrecognized IP in Eastern Europe. Lock my account.", "security_compromise"),
    ("收到异地短信登录提醒，我本人并未登录，怀疑密码被撞库泄露！", "security_compromise"),
    ("We have 2,000 employees. We need a formal quotation for site-wide licensing.", "sales_inquiry"),
    ("我们集团计划采购 100 套企业私有化部署授权，请安排销售总监对接商务报价。", "sales_inquiry"),
    ("Under GDPR Article 17, delete all my personal records from your production database.", "gdpr_compliance"),
    ("依据个人信息保护合规政策，要求永久擦除我在贵平台的所有历史日志和隐私信息。", "gdpr_compliance"),
    ("I switched to a new phone and cannot receive the 2FA SMS authentication code.", "account_login_issue"),
    ("账号绑定的手机号停用了，收不到登录验证码，怎么通过人工申诉找回？", "account_login_issue"),
]

# DISJOINT TEMPLATES FOR CALIBRATION (15%)
CALIB_TRIAGE_TEMPLATES = [
    ("Duplicate billing occurred for invoice ID {num}. Refund the second charge immediately.", "billing_duplicate"),
    ("同一笔订单微信和支付宝各自扣款了一次，总共付了双倍金额，请核实退款。", "billing_duplicate"),
    ("The software is totally unusable on my laptop, I demand a refund according to policy.", "billing_refund"),
    ("误操作付款购买了服务，还没开始使用，希望客服支持退还费用。", "billing_refund"),
    ("Cancel my pro membership immediately. I do not want any auto renewals.", "subscription_cancel"),
    ("取消我的年费订阅，下月到期后不要再扣我的信用卡了。", "subscription_cancel"),
    ("We want to switch our active plan from the $99 plan to the $19 plan.", "subscription_downgrade"),
    ("业务调整，团队只需要最基础的功能席位，请求降级现有方案。", "subscription_downgrade"),
    ("The export button does nothing when clicked on Safari 18.", "tech_bug"),
    ("在火狐浏览器下上传附件进度条卡在 99% 不动，无法完成提交。", "tech_bug"),
    ("All API nodes are unreachable from North America. Complete service outage.", "tech_outage"),
    ("数据库主从同步中断，核心接口全部报 500 内部错误，生产服务瘫痪！", "tech_outage"),
    ("Someone unauthorized changed the primary email address on our corporate account.", "security_compromise"),
    ("账户资金有未授权的提现记录，疑似 API Key 密钥被黑客窃取！", "security_compromise"),
    ("We are a research institute looking for non-profit enterprise discounts.", "sales_inquiry"),
    ("学校科研团队想批量采购 30 套专业版，请问是否有教育专属采购折扣？", "sales_inquiry"),
    ("Please export a complete JSON dump of all my personal data per GDPR compliance.", "gdpr_compliance"),
    ("申请行使被遗忘权，清空服务器上保存的本人身份信息与行为记录。", "gdpr_compliance"),
    ("Forgot my login master password and reset email never arrives.", "account_login_issue"),
    ("密保问题忘记了，重置密码链接打不开，一直被卡在登录界面。", "account_login_issue"),
]

# DISJOINT TEMPLATES FOR TEST (15%) - COMPLETELY NOVEL NATURAL LANGUAGE PATTERNS
TEST_TRIAGE_TEMPLATES = [
    ("Why was my Amex billed $120 twice yesterday morning? Reverse one transaction.", "billing_duplicate"),
    ("对账发现本期账单同一笔订阅出现了两次扣费流水，麻烦把多扣的一笔退回原账户。", "billing_duplicate"),
    ("Not satisfied with the service quality, please issue a credit refund to my balance.", "billing_refund"),
    ("买了发现根本不兼容我们旧系统，按约定条款申请原路退款处理。", "billing_refund"),
    ("Stop charging me! I have moved to another tool, discontinue our contract immediately.", "subscription_cancel"),
    ("我不打算继续使用了，请客服直接在后台帮我彻底关停会员账户。", "subscription_cancel"),
    ("Our startup is downsizing. What is the process to move down to the Community tier?", "subscription_downgrade"),
    ("预算吃紧，需要把当前的企业级高阶配额调低到基础档位以节省开销。", "subscription_downgrade"),
    ("The PDF generator misaligns Chinese fonts and cuts off the footer on Linux.", "tech_bug"),
    ("移动端每次下拉刷新都有概率直接闪退，复现路径已抓取日志。", "tech_bug"),
    ("CRITICAL INCIDENT: DNS resolution fails globally, customers cannot reach store front.", "tech_outage"),
    ("海外 CDN 节点全部失联，大量用户反馈页面加载超时打不开网站。", "tech_outage"),
    ("Security breach: employee credentials were discovered in a public data paste.", "security_compromise"),
    ("安全监控发现大量高频异地 IP 尝试暴力破解主管理员账号密码，请紧急处置！", "security_compromise"),
    ("Can you provide an enterprise MSA and schedule a vendor security assessment call?", "sales_inquiry"),
    ("准备向公司采购部立项引入该系统，需要销售经理提供标准化采购合同与对公账户信息。", "sales_inquiry"),
    ("Formally exercising Article 17 right to erasure. Confirm permanent deletion.", "gdpr_compliance"),
    ("请出具个人数据已被物理彻底删除的书面合规确认函。", "gdpr_compliance"),
    ("I am locked out because authenticator app app crashed and backup codes are lost.", "account_login_issue"),
    ("输入三次错误密码后账户被临时锁定了，提示需联系系统管理员解锁。", "account_login_issue"),
]

# ==============================================================================
# Domain 2: Tool Routing Templates (Disjoint)
# ==============================================================================
TOOL_CATALOG = [
    ("calculator", "精确数学四则运算、汇率换算、复杂公式求值与几何度量"),
    ("web_search", "在互联网中检索实时新闻、公网百科资讯与最新学术报告"),
    ("bash_runner", "在终端环境中执行 Shell 命令、文件查找与系统进程管理"),
    ("database_query", "通过 SQL 查询或更新企业内部关系型业务数据库表"),
    ("weather_api", "查询各大主要城市未来几天的气温、湿度、降水与天气预报"),
    ("calendar_scheduler", "创建、同步、修改或查询个人与团队的日程会议安排"),
    ("code_interpreter", "在沙盒中运行 Python 脚本进行数据建模、清洗与图表绘制"),
]

TRAIN_TOOL_TEMPLATES = [
    ("计算 {a} 的平方加上 {b} 的立方等于多少？", "calculator"),
    ("Calculate the derivative of x^{b} at x={a}.", "calculator"),
    ("搜索今天关于 {topic} 的最新科技新闻报道。", "web_search"),
    ("Search Google for recent papers on {topic} published in 2026.", "web_search"),
    ("在 Linux 终端查看 /var/log/ 目录下占用空间最大的文件。", "bash_runner"),
    ("Find all running python processes using bash and list their PIDs.", "bash_runner"),
    ("从 users 表中查询注册时间在去年且消费大于 {a} 的客户名单。", "database_query"),
    ("Run SQL query to compute average transaction size by region.", "database_query"),
    ("查一下明天 {city} 的最高气温和穿衣指数。", "weather_api"),
    ("What is the precipitation forecast for {city} this weekend?", "weather_api"),
    ("帮我安排下周二下午 3 点与设计团队的技术评审会议。", "calendar_scheduler"),
    ("Schedule a 45-minute sync meeting with client {topic} tomorrow.", "calendar_scheduler"),
    ("用 Python 的 pandas 和 matplotlib 绘制这组数据的散点图。", "code_interpreter"),
    ("Write and execute a Python script to train a logistic regression model.", "code_interpreter"),
]

CALIB_TOOL_TEMPLATES = [
    ("求 {a} 乘以 {b} 的倒数再加上 {c} 的数值。", "calculator"),
    ("Calculate compound interest on ${a} at {b}% over {c} periods.", "calculator"),
    ("检索互联网上关于 {topic} 的技术白皮书和官方文档。", "web_search"),
    ("Look up the Wikipedia biography and background of {topic}.", "web_search"),
    ("用终端命令统计 access.log 文件中 404 状态码出现的次数。", "bash_runner"),
    ("Check free disk space and memory usage via command line.", "bash_runner"),
    ("统计 orders 表中按支付渠道划分的退单率与金额。", "database_query"),
    ("Select top {b} highest grossing products from inventory table.", "database_query"),
    ("看看明天去 {city} 出差需要带雨伞吗，降雨概率是多少？", "weather_api"),
    ("Is it expected to snow in {city} next Monday?", "weather_api"),
    ("在日程表里把今天下午的会议延后一个小时举行。", "calendar_scheduler"),
    ("Block out 2 hours on my calendar next Friday for deep work.", "calendar_scheduler"),
    ("写一段 Python 代码把这个 CSV 文件里的缺失值用均值填充。", "code_interpreter"),
    ("Generate a correlation heatmap using seaborn in a Python sandbox.", "code_interpreter"),
]

TEST_TOOL_TEMPLATES = [
    ("求直角三角形两直角边为 {a} 和 {b} 时的斜边长与内切圆半径。", "calculator"),
    ("Evaluate the expression ({a} * {b} + {c}) / 7.5 precisely.", "calculator"),
    ("在全网搜索 {topic} 领域的最新技术突破和融资消息。", "web_search"),
    ("Search the web for user reviews and benchmark comparisons of {topic}.", "web_search"),
    ("在后台查看端口 8080 是否有进程在监听并查看其连接数。", "bash_runner"),
    ("Execute a shell script to compress all .log files older than 30 days.", "bash_runner"),
    ("在数据库中关联 orders 表和 users 表，找出复购率最高的用户群组。", "database_query"),
    ("Update customer loyalty points where total_spent > {a} in SQL.", "database_query"),
    ("查一下 {city} 接下来三天的风向、风力级别和空气污染指数。", "weather_api"),
    ("Get the current humidity, visibility, and UV index in {city}.", "weather_api"),
    ("查看我下周三上午是否有时间空闲可以插入一个临时面试？", "calendar_scheduler"),
    ("Cancel my upcoming calendar sync with stakeholder {topic}.", "calendar_scheduler"),
    ("写一段 Python 程序利用 scipy 计算这组高维数据的奇异值分解并画图。", "code_interpreter"),
    ("Run a Monte Carlo simulation in Python to estimate option volatility.", "code_interpreter"),
]

# ==============================================================================
# Domain 3: Dialogue Feedback & Ambiguity (Anti-example coverage)
# ==============================================================================
DIALOGUE_CATALOG = [
    ("confirm_satisfied", "用户表示明确满意、认可方案或指令完成 / Customer clearly satisfied"),
    ("ambiguous_clarify", "态度模糊或语义不明确需进一步提问确认 / Ambiguous needs clarification"),
    ("negative_reject", "用户明确否定、拒绝方案或表达强烈不满 / Customer explicitly dissatisfied"),
]

DIALOGUE_SAMPLES_TEST = [
    ("“这个方案满意吗？”——“行，可以。”", "confirm_satisfied"),
    ("“修改后的界面满意吗？”——“没问题，挺好的，就按这个来。”", "confirm_satisfied"),
    ("“您看这样处理行不行？”——“嗯，我看还行吧，不过我也说不准，大家再商量商量？”", "ambiguous_clarify"),
    ("“关于昨天的报价您考虑得如何？”——“还行吧，但是我们内部还在走流程，下周再说。”", "ambiguous_clarify"),
    ("“对本次服务结果满意吗？”——“太差了，完全不合逻辑，全部推倒重做！”", "negative_reject"),
    ("“可以开始部署吗？”——“不行，坚决不同意，立刻终止操作。”", "negative_reject"),
]

TOPICS = ["量子计算", "端侧AI模型", "向量数据库", "Rust异步并发", "大模型对齐", "混合专家网络MoE"]
CITIES = ["北京", "上海", "广州", "深圳", "成都", "New York", "London", "Tokyo", "Berlin"]


def build_sample(template_pool, catalog_pool, domain_name, question_str):
    tmpl, target_key = random.choice(template_pool)
    text = tmpl.format(
        num=random.randint(1000, 9999),
        money=random.randint(20, 500),
        a=random.randint(10, 200),
        b=random.randint(2, 30),
        c=random.randint(2, 10),
        topic=random.choice(TOPICS),
        city=random.choice(CITIES)
    )
    target_tuple = next(t for t in catalog_pool if t[0] == target_key)
    target_option_str = f"{target_tuple[0]}: {target_tuple[1]}"
    
    # Dynamic K (4 to 8)
    k = min(len(catalog_pool), random.randint(4, 7))
    distractors = [f"{t[0]}: {t[1]}" for t in catalog_pool if t[0] != target_key]
    selected_distractors = random.sample(distractors, k - 1)
    
    insert_pos = random.randint(0, len(selected_distractors))
    selected_distractors.insert(insert_pos, target_option_str)
    
    return {
        "domain": domain_name,
        "state": text,
        "question": question_str,
        "candidates": selected_distractors,
        "target_idx": insert_pos,
        "target_key": target_key,
    }


def generate_datasets():
    print("Generating Template-Disjoint Datasets V3...")
    
    # 1. Train Split (70%, ~1400 samples)
    train_data = []
    for i in range(1400):
        if random.random() < 0.55:
            s = build_sample(TRAIN_TRIAGE_TEMPLATES, TRIAGE_CATALOG, "triage", "Which department or category best fits the customer ticket?")
        else:
            s = build_sample(TRAIN_TOOL_TEMPLATES, TOOL_CATALOG, "tool", "Which system tool should the AI agent invoke to complete this request?")
        s["id"] = f"train_v3_{i:05d}"
        train_data.append(s)
        
    # 2. Calibration Split (15%, ~300 samples)
    calib_data = []
    for i in range(300):
        if random.random() < 0.55:
            s = build_sample(CALIB_TRIAGE_TEMPLATES, TRIAGE_CATALOG, "triage", "Which department or category best fits the customer ticket?")
        else:
            s = build_sample(CALIB_TOOL_TEMPLATES, TOOL_CATALOG, "tool", "Which system tool should the AI agent invoke to complete this request?")
        s["id"] = f"calib_v3_{i:05d}"
        calib_data.append(s)
        
    # 3. Test Split (15%, ~300 samples)
    test_data = []
    for i in range(270):
        if random.random() < 0.55:
            s = build_sample(TEST_TRIAGE_TEMPLATES, TRIAGE_CATALOG, "triage", "Which department or category best fits the customer ticket?")
        else:
            s = build_sample(TEST_TOOL_TEMPLATES, TOOL_CATALOG, "tool", "Which system tool should the AI agent invoke to complete this request?")
        s["id"] = f"test_v3_{i:05d}"
        test_data.append(s)
        
    # Add anti-example dialogue feedback items into test split
    for i, (dial_text, target_k) in enumerate(DIALOGUE_SAMPLES_TEST * 5):
        target_t = next(t for t in DIALOGUE_CATALOG if t[0] == target_k)
        target_opt = f"{target_t[0]}: {target_t[1]}"
        distract = [f"{t[0]}: {t[1]}" for t in DIALOGUE_CATALOG if t[0] != target_k]
        pos = random.randint(0, len(distract))
        distract.insert(pos, target_opt)
        test_data.append({
            "domain": "dialogue_feedback",
            "state": dial_text,
            "question": "用户对待当前方案的态度属于哪种意图类型？",
            "candidates": distract,
            "target_idx": pos,
            "target_key": target_k,
            "id": f"test_v3_dialogue_{i:03d}"
        })
        
    train_path = OUTPUT_DIR / "train_v3.json"
    calib_path = OUTPUT_DIR / "calib_v3.json"
    test_path = OUTPUT_DIR / "test_v3.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print(f"Train V3: {len(train_data)} samples -> {train_path}")
    print(f"Calib V3: {len(calib_data)} samples -> {calib_path}")
    print(f"Test V3:  {len(test_data)} samples -> {test_path}")

if __name__ == "__main__":
    generate_datasets()
