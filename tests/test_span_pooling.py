# -*- coding: utf-8 -*-
"""
Unit test for option span mean pooling and gradient flow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model
from src.tokenizer_utils import encode_decision_batch
from src.modeling_s1 import S1DecisionModel


def test_span_extraction_and_pooling():
    base_model_path = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    state = "Customer says: My account has been locked after 3 failed password attempts."
    question = "Which department should handle this?"
    options = [
        "security_auth: reset password or 2FA token",
        "billing_payment: invoice questions and payments",
        "general_inquiry: business hours and locations",
    ]
    
    batch = encode_decision_batch(
        tokenizer=tokenizer,
        states=[state],
        questions=[question],
        options_per_sample=[options],
        targets=[0],
        device="cpu",
    )
    
    assert "option_spans" in batch
    spans = batch["option_spans"][0]
    input_ids = batch["input_ids"][0]
    
    print("\nVerifying Option Spans:")
    for k in range(len(options)):
        s, e = spans[k, 0].item(), spans[k, 1].item()
        assert e > s, f"Option {k} span invalid: [{s}, {e}]"
        decoded_span = tokenizer.decode(input_ids[s:e], skip_special_tokens=True).strip()
        print(f"  Option {k} Span [{s}:{e}] -> '{decoded_span}'")
        # Verify span text contains keywords from options[k]
        first_word = options[k].split(":")[0]
        assert first_word in decoded_span, f"Expected '{first_word}' in decoded span: '{decoded_span}'"
        
    print("[PASS] Span extraction verified!")


def test_span_pooling_forward_backward():
    base_model_path = "E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    config = AutoConfig.from_pretrained(base_model_path)
    hidden_dim = getattr(config, "hidden_size", 896)
    
    backbone = AutoModel.from_pretrained(base_model_path, config=config, torch_dtype=torch.float32)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    
    model = S1DecisionModel(backbone, hidden_dim=hidden_dim, num_heads=4, num_inter_layers=1)
    
    state = "User requests refund."
    question = "Action?"
    options = ["refund", "cancel"]
    
    batch = encode_decision_batch(
        tokenizer=tokenizer,
        states=[state],
        questions=[question],
        options_per_sample=[options],
        targets=[0],
        device="cpu",
    )
    
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        marker_indices=batch["marker_indices"],
        marker_mask=batch["marker_mask"],
        option_spans=batch["option_spans"],
    )
    
    loss = out["probs"][0, 0]
    loss.backward()
    
    print(f"Forward output logits shape: {out['logits'].shape}")
    print("[PASS] Span pooling forward & backward passed!")


if __name__ == "__main__":
    test_span_extraction_and_pooling()
    test_span_pooling_forward_backward()
    print("\n[ALL SPAN POOLING TESTS PASSED!]")
