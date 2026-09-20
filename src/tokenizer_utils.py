# -*- coding: utf-8 -*-
"""
Tokenizer and Prompt Formatting Utilities for S1 Decision Models.
Formats state, questions, and dynamic candidate options with option markers,
and extracts marker tensor coordinates for batch processing.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import PreTrainedTokenizer


DEFAULT_MARKER_TOKEN = "<|fim_pad|>"


def format_decision_prompt(
    state: str,
    question: str,
    options: List[str],
    marker: str = DEFAULT_MARKER_TOKEN,
) -> str:
    """
    Format state, instructions, and candidate options into a single prompt string.
    Example:
      [状态]: 用户在结账页面点击了取消。
      [决策问题]: 用户的意图是什么？
      [候选选项]:
      [MASK] 放弃购买: 用户离开不再购买
      [MASK] 暂存购物车: 用户想稍后再买
      [MASK] 咨询客服: 遇到支付异常需要求助
    """
    state_str = (state or "").strip()
    q_str = (question or "").strip()
    
    options_formatted = []
    for opt in options:
        opt_clean = opt.strip()
        options_formatted.append(f"{marker} {opt_clean}")
    options_str = "\n".join(options_formatted)
    
    prompt = (
        f"[状态说明]: {state_str}\n"
        f"[决策指令]: {q_str}\n"
        f"[候选动作]:\n{options_str}"
    )
    return prompt


def encode_decision_batch(
    tokenizer: PreTrainedTokenizer,
    batch_prompts: List[str],
    options_per_sample: List[List[str]],
    targets: Optional[List[int]] = None,
    marker: str = DEFAULT_MARKER_TOKEN,
    max_length: int = 512,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes batch of prompts, locates marker token positions,
    and returns tensors ready for S1DecisionModel.
    """
    # Ensure marker token is known
    marker_token_id = tokenizer.convert_tokens_to_ids(marker)
    if marker_token_id is None or marker_token_id == tokenizer.unk_token_id:
        # Fallback to mask_token or eos_token if marker not explicitly present
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            marker_token_id = tokenizer.mask_token_id
        else:
            marker_token_id = tokenizer.eos_token_id

    encoded = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    batch_size = input_ids.shape[0]
    
    # Calculate max number of candidate options in this batch
    max_k = max(len(opts) for opts in options_per_sample)
    
    marker_indices = torch.zeros((batch_size, max_k), dtype=torch.long, device=device)
    marker_mask = torch.zeros((batch_size, max_k), dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        num_options = len(options_per_sample[b])
        # Find all positions of marker_token_id in sample b
        token_positions = (input_ids[b] == marker_token_id).nonzero(as_tuple=True)[0]
        
        # Take at most num_options positions
        valid_count = min(len(token_positions), num_options)
        for k in range(valid_count):
            marker_indices[b, k] = token_positions[k]
            marker_mask[b, k] = True
            
        # In case fewer markers were found than options (e.g. truncated), fill remaining with last token
        if valid_count < num_options:
            for k in range(valid_count, num_options):
                marker_indices[b, k] = input_ids.shape[1] - 1
                marker_mask[b, k] = True

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_indices": marker_indices,
        "marker_mask": marker_mask,
    }
    
    if targets is not None:
        result["targets"] = torch.tensor(targets, dtype=torch.long, device=device)
        
    return result
