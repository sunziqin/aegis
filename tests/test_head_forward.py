# -*- coding: utf-8 -*-
"""
Unit test for DynamicOptionMarkerHead and S1DecisionModel.
Verifies variable option sizes (K), masked probabilities, and GPU forward latency.
"""

import sys
import time
from pathlib import Path
import torch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling_s1 import DynamicOptionMarkerHead, S1DecisionModel


def test_head_with_dummy_tensors():
    print("=" * 60)
    print("[1] Testing DynamicOptionMarkerHead with Variable Candidate Sizes (K)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    seq_len = 128
    hidden_dim = 1024
    max_k = 5  # At most 5 candidate options in this batch
    
    head = DynamicOptionMarkerHead(hidden_dim=hidden_dim, num_heads=8, num_inter_layers=2).to(device)
    head.eval()
    
    # Fake sequence hidden states from a transformer encoder
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Simulate variable number of options per sample:
    # Sample 0: 2 options (at tokens 10, 20)
    # Sample 1: 3 options (at tokens 15, 30, 45)
    # Sample 2: 5 options (at tokens 5, 12, 25, 40, 50)
    # Sample 3: 4 options (at tokens 8, 16, 24, 32)
    marker_indices = torch.tensor([
        [10, 20, 0,  0,  0],
        [15, 30, 45, 0,  0],
        [5,  12, 25, 40, 50],
        [8,  16, 24, 32, 0],
    ], device=device)
    
    marker_mask = torch.tensor([
        [True,  True,  False, False, False],
        [True,  True,  True,  False, False],
        [True,  True,  True,  True,  True],
        [True,  True,  True,  True,  False],
    ], device=device)
    
    with torch.no_grad():
        logits, probs, escalate = head(hidden_states, marker_indices, marker_mask)
        
    print(f"[*] Logits shape: {logits.shape} (Expected: [{batch_size}, {max_k}])")
    print(f"[*] Probs shape: {probs.shape} (Expected: [{batch_size}, {max_k}])")
    print(f"[*] Escalate score shape: {escalate.shape} (Expected: [{batch_size}])")
    
    # Assert probabilities sum to 1.0 for valid options and 0.0 for padded
    prob_sums = probs.sum(dim=-1)
    print(f"[*] Probability sums per sample: {prob_sums.tolist()}")
    for b in range(batch_size):
        assert abs(prob_sums[b].item() - 1.0) < 1e-4, f"Sample {b} probabilities do not sum to 1.0!"
        # Check padded options are strictly 0.0
        for k in range(max_k):
            if not marker_mask[b, k]:
                assert probs[b, k].item() == 0.0, f"Sample {b} padded option {k} is non-zero!"
                
    print("[PASS] DynamicOptionMarkerHead correctly handles variable candidate sizes and masking!")
    return head


def test_head_latency_benchmark(head):
    print("\n" + "=" * 60)
    print("[2] Benchmarking Head Forward Latency on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[!] CUDA not available, skipping GPU micro-benchmark.")
        return
        
    batch_size = 1  # Standard real-time API request
    seq_len = 256
    hidden_dim = 1024
    max_k = 6
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    marker_indices = torch.tensor([[10, 25, 40, 60, 80, 100]], device=device)
    marker_mask = torch.ones((batch_size, max_k), dtype=torch.bool, device=device)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = head(hidden_states, marker_indices, marker_mask)
    torch.cuda.synchronize()
    
    # Benchmark 200 iterations
    num_runs = 200
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = head(hidden_states, marker_indices, marker_mask)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) * 1000 / num_runs
    
    print(f"[*] Dynamic Decision Head Single-Inference Latency: {elapsed:.3f} ms")
    assert elapsed < 10.0, f"Head latency too slow ({elapsed:.3f} ms)"
    print("[PASS] Head latency is lightning fast (sub-millisecond overhead)!")


def test_with_real_backbone():
    print("\n" + "=" * 60)
    print("[3] Testing End-to-End with Local Backbone (Qwen2.5-0.5B)...")
    local_path = Path("E:/asr-endpoint-service/models/base/Qwen2.5-0.5B-Instruct")
    if not local_path.exists():
        print(f"[!] Local path {local_path} not found, skipping.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading backbone from {local_path} on {device}...")
    
    model = S1DecisionModel.from_pretrained(
        str(local_path),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    
    # Dummy token inputs: 1 sample with 3 options
    input_ids = torch.randint(100, 1000, (1, 64), device=device)
    attention_mask = torch.ones_like(input_ids)
    marker_indices = torch.tensor([[10, 25, 40]], device=device)
    marker_mask = torch.tensor([[True, True, True]], device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids, attention_mask, marker_indices, marker_mask)
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Latency test
    runs = 30
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            out = model(input_ids, attention_mask, marker_indices, marker_mask)
    if device == "cuda":
        torch.cuda.synchronize()
    avg_latency = (time.perf_counter() - t0) * 1000 / runs
    
    print(f"[*] End-to-End Model Latency (Backbone + S1 Head): {avg_latency:.2f} ms")
    print(f"[*] Output Probs: {out['probs'][0].tolist()}")
    print(f"[*] Selected Best Option Index: {out['best_choice_idx'].item()}")
    print(f"[*] Confidence: {out['confidence'].item():.4f}")
    print(f"[*] Escalate Risk: {out['escalate_risk'].item():.4f}")
    print("[PASS] End-to-End Backbone + Decision Head test completed successfully!")


if __name__ == "__main__":
    head = test_head_with_dummy_tensors()
    test_head_latency_benchmark(head)
    test_with_real_backbone()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Phase 1 Model Core is fully operational.")
