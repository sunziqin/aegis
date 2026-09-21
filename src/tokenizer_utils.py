# -*- coding: utf-8 -*-
"""
Tokenizer and Prompt Formatting Utilities for S1 Decision Models.
Formats state, questions, and dynamic candidate options with option markers,
implements safe left-truncation to protect candidate options and markers,
sanitizes prompt inputs against control character injection,
and extracts marker tensor coordinates for batch processing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import PreTrainedTokenizer


DEFAULT_MARKER_TOKEN = "<|fim_pad|>"


def sanitize_text(text: str, marker: str = DEFAULT_MARKER_TOKEN) -> str:
    """
    Sanitize raw user text to prevent special token injection attacks
    that could corrupt marker coordinate alignment.
    """
    if not text:
        return ""
    # Strip any accidental or malicious marker tokens and end-of-text tokens
    cleaned = (
        str(text)
        .replace(marker, " ")
        .replace("<|endoftext|>", " ")
        .replace("<|im_start|>", " ")
        .replace("<|im_end|>", " ")
        .replace("[MASK]", " ")
    )
    return cleaned.strip()


def format_decision_prompt(
    state: str,
    question: str,
    options: List[str],
    marker: str = DEFAULT_MARKER_TOKEN,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 512,
) -> str:
    """
    Format state, instructions, and candidate options into a single prompt string.
    If tokenizer is provided, applies smart Left-Truncation to the state:
    the candidate options and question instructions are ALWAYS preserved,
    while older state tokens are safely truncated from the left.
    """
    state_clean = sanitize_text(state, marker=marker)
    q_clean = sanitize_text(question, marker=marker)
    
    options_formatted = []
    for opt in options:
        opt_clean = sanitize_text(opt, marker=marker)
        options_formatted.append(f"{marker} {opt_clean}")
    options_str = "\n".join(options_formatted)
    
    head_str = f"\n[决策指令]: {q_clean}\n[候选动作]:\n{options_str}"
    
    if tokenizer is not None:
        head_tokens = tokenizer.encode(head_str, add_special_tokens=False)
        room = max_length - len(head_tokens) - 16
        if room < 16:
            raise ValueError(
                f"Candidate options ({len(options)} choices, {len(head_tokens)} tokens) "
                f"exceed max_length={max_length} token budget. Please increase max_length."
            )
        state_tokens = tokenizer.encode(f"[状态说明]: {state_clean}", add_special_tokens=False)
        if len(state_tokens) > room:
            # Left-truncate state: discard earliest context, retain freshest context
            state_tokens = state_tokens[-room:]
            state_clean_truncated = tokenizer.decode(state_tokens, skip_special_tokens=True)
            return f"{state_clean_truncated}{head_str}"
            
    return f"[状态说明]: {state_clean}{head_str}"


def encode_decision_batch(
    tokenizer: PreTrainedTokenizer,
    batch_prompts: Optional[List[str]] = None,
    options_per_sample: Optional[List[List[str]]] = None,
    states: Optional[List[str]] = None,
    questions: Optional[List[str]] = None,
    targets: Optional[List[int]] = None,
    marker: str = DEFAULT_MARKER_TOKEN,
    max_length: int = 512,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes batch of prompts, locates marker token positions,
    and returns tensors ready for S1DecisionModel.
    Guarantees that candidate markers are NEVER truncated away.
    """
    marker_token_id = tokenizer.convert_tokens_to_ids(marker)
    if marker_token_id is None or marker_token_id == tokenizer.unk_token_id:
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            marker_token_id = tokenizer.mask_token_id
        else:
            marker_token_id = tokenizer.eos_token_id

    # If structured states/questions provided, build prompts with smart left-truncation
    if states is not None and questions is not None and options_per_sample is not None:
        batch_prompts = []
        for b in range(len(states)):
            p = format_decision_prompt(
                state=states[b],
                question=questions[b],
                options=options_per_sample[b],
                marker=marker,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            batch_prompts.append(p)
    elif batch_prompts is None:
        raise ValueError("Either batch_prompts or (states, questions, options_per_sample) must be provided.")

    if options_per_sample is None:
        raise ValueError("options_per_sample must be provided.")

    batch_size = len(batch_prompts)
    
    encoded = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    seq_len = input_ids.shape[1]
    
    max_k = max(len(opts) for opts in options_per_sample)
    marker_indices = torch.zeros((batch_size, max_k), dtype=torch.long, device=device)
    marker_mask = torch.zeros((batch_size, max_k), dtype=torch.bool, device=device)
    option_spans = torch.zeros((batch_size, max_k, 2), dtype=torch.long, device=device)
    
    for b in range(batch_size):
        num_options = len(options_per_sample[b])
        token_positions = (input_ids[b] == marker_token_id).nonzero(as_tuple=True)[0]
        
        if len(token_positions) < num_options:
            raise ValueError(
                f"Sample {b}: Expected {num_options} candidate markers for options, "
                f"but only found {len(token_positions)}. This indicates options were truncated "
                f"or corrupted. Max length is {max_length}."
            )
            
        non_pad = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
        seq_end = non_pad[-1].item() + 1 if len(non_pad) > 0 else seq_len

        for k in range(num_options):
            s_pos = token_positions[k].item()
            if k < num_options - 1:
                e_pos = token_positions[k + 1].item()
            else:
                e_pos = seq_end

            opt_start = s_pos + 1 if e_pos > s_pos + 1 else s_pos
            option_spans[b, k, 0] = opt_start
            option_spans[b, k, 1] = e_pos
            marker_indices[b, k] = s_pos
            marker_mask[b, k] = True

    # Construct 4D Bidirectional Attention Mask
    mask_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    mask_4d = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=mask_dtype, device=device)
    for b in range(batch_size):
        pad_idx = (attention_mask[b] == 0).nonzero(as_tuple=True)[0]
        if len(pad_idx) > 0:
            mask_4d[b, 0, :, pad_idx] = -1e4
            mask_4d[b, 0, pad_idx, :] = -1e4

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bidirectional_mask_4d": mask_4d,
        "marker_indices": marker_indices,
        "marker_mask": marker_mask,
        "option_spans": option_spans,
    }
    
    if targets is not None:
        result["targets"] = torch.tensor(targets, dtype=torch.long, device=device)
        
    return result


def format_multi_query_prompt(
    state: str,
    queries: Dict[str, Tuple[str, List[str]]],
    marker: str = DEFAULT_MARKER_TOKEN,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 1024,
) -> str:
    """
    Format state and multiple heterogeneous decision queries into a single unified prompt.
    Protects all query instructions and candidate option markers with intelligent left-truncation.
    """
    state_clean = sanitize_text(state, marker=marker)
    
    sections = []
    for q_name, (q_text, opts) in queries.items():
        q_clean = sanitize_text(q_text, marker=marker)
        opts_formatted = [f"{marker} {sanitize_text(opt, marker=marker)}" for opt in opts]
        opts_str = "\n".join(opts_formatted)
        sections.append(f"\n[决策指令: {q_name}]: {q_clean}\n[候选动作: {q_name}]:\n{opts_str}")
        
    queries_str = "\n".join(sections)
    
    if tokenizer is not None:
        queries_tokens = tokenizer.encode(queries_str, add_special_tokens=False)
        room = max_length - len(queries_tokens) - 16
        if room < 16:
            raise ValueError(
                f"Multi-queries ({len(queries)} queries, {len(queries_tokens)} tokens) "
                f"exceed max_length={max_length} token budget. Please increase max_length."
            )
        state_tokens = tokenizer.encode(f"[状态说明]: {state_clean}", add_special_tokens=False)
        if len(state_tokens) > room:
            state_tokens = state_tokens[-room:]
            state_clean_truncated = tokenizer.decode(state_tokens, skip_special_tokens=True)
            return f"{state_clean_truncated}{queries_str}"
            
    return f"[状态说明]: {state_clean}{queries_str}"


def encode_multi_query_batch(
    tokenizer: PreTrainedTokenizer,
    states: List[str],
    queries_per_sample: List[Dict[str, Tuple[str, List[str]]]],
    marker: str = DEFAULT_MARKER_TOKEN,
    max_length: int = 1024,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Tokenizes a batch of multi-query prompts and locates marker token coordinates
    for each query in the prompt, enabling single-forward-pass parallel evaluation.
    """
    marker_token_id = tokenizer.convert_tokens_to_ids(marker)
    if marker_token_id is None or marker_token_id == tokenizer.unk_token_id:
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            marker_token_id = tokenizer.mask_token_id
        else:
            marker_token_id = tokenizer.eos_token_id

    batch_size = len(states)
    batch_prompts = []
    for b in range(batch_size):
        p = format_multi_query_prompt(
            state=states[b],
            queries=queries_per_sample[b],
            marker=marker,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        batch_prompts.append(p)

    encoded = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    seq_len = input_ids.shape[1]

    # Query names from sample 0
    query_names = list(queries_per_sample[0].keys())
    query_markers = {}

    for q_name in query_names:
        max_k_q = max(len(queries_per_sample[b][q_name][1]) for b in range(batch_size))
        marker_indices_q = torch.zeros((batch_size, max_k_q), dtype=torch.long, device=device)
        marker_mask_q = torch.zeros((batch_size, max_k_q), dtype=torch.bool, device=device)
        option_spans_q = torch.zeros((batch_size, max_k_q, 2), dtype=torch.long, device=device)
        query_markers[q_name] = {
            "indices": marker_indices_q,
            "mask": marker_mask_q,
            "spans": option_spans_q,
        }

    for b in range(batch_size):
        token_positions = (input_ids[b] == marker_token_id).nonzero(as_tuple=True)[0]
        total_expected_markers = sum(len(opts) for _, opts in queries_per_sample[b].values())
        if len(token_positions) < total_expected_markers:
            raise ValueError(
                f"Sample {b}: Expected {total_expected_markers} markers across {len(query_names)} queries, "
                f"but found {len(token_positions)}. Tokens truncated or corrupted."
            )
        
        non_pad = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
        seq_end = non_pad[-1].item() + 1 if len(non_pad) > 0 else seq_len

        offset = 0
        for q_name, (q_text, opts) in queries_per_sample[b].items():
            num_opts = len(opts)
            for k in range(num_opts):
                global_idx = offset + k
                s_pos = token_positions[global_idx].item()
                if global_idx < total_expected_markers - 1:
                    e_pos = token_positions[global_idx + 1].item()
                else:
                    e_pos = seq_end

                opt_start = s_pos + 1 if e_pos > s_pos + 1 else s_pos
                query_markers[q_name]["indices"][b, k] = s_pos
                query_markers[q_name]["mask"][b, k] = True
                query_markers[q_name]["spans"][b, k, 0] = opt_start
                query_markers[q_name]["spans"][b, k, 1] = e_pos
            offset += num_opts


    # Construct 4D Bidirectional Attention Mask
    mask_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    mask_4d = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=mask_dtype, device=device)
    for b in range(batch_size):
        pad_idx = (attention_mask[b] == 0).nonzero(as_tuple=True)[0]
        if len(pad_idx) > 0:
            mask_4d[b, 0, :, pad_idx] = -1e4
            mask_4d[b, 0, pad_idx, :] = -1e4

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bidirectional_mask_4d": mask_4d,
        "query_markers": query_markers,
    }

