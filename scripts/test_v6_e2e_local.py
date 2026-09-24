# -*- coding: utf-8 -*-
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from open_s1 import AegisRouter

def test_local_e2e():
    print("[*] Loading fresh V6 weights into AegisRouter...")
    router = AegisRouter.load(
        str(repo_root / "output" / "s1_model_v6"),
        base_model_path="E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct",
        fuse_lora=True,
    )
    
    test_state = "用户申请跨国大额转账，系统检测到异地登录但已完成人脸验证。"
    test_question = "请选择处置动作"
    test_candidates = [
        "直接放行并提交SWIFT",
        "转入反洗钱人工复核",
        "永久冻结外汇结算",
        "要求线下网点重新核验"
    ]
    
    print("\n[*] Running test decision...")
    res = router.decide(
        state=test_state,
        question=test_question,
        candidates=test_candidates,
    )
    
    print("=" * 60)
    print(f"Selected Option:      {res['selected_option']}")
    print(f"Confidence:           {res['confidence']:.2%}")
    print(f"Can Act (Autonomous): {res['can_act']}")
    print(f"Conformal Verdict:    {res['conformal_verdict']}")
    print(f"Prediction Set:       {res['prediction_set']}")
    print(f"Escalate Risk Gate:   {res['escalate_risk']:.4f}")
    print("=" * 60)
    print("[PASS] End-to-end inference verification successful!")

if __name__ == "__main__":
    test_local_e2e()
