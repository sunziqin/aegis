# -*- coding: utf-8 -*-
"""
Tokenizer and Prompt Formatting Utilities for S1 Decision Models.
Formats state, questions, and dynamic candidate options with option markers,
implements safe left-truncation to protect candidate options and markers,
sanitizes prompt inputs against control character injection,
and extracts marker tensor coordinates for batch processing.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


DEFAULT_MARKER_TOKEN = "<|fim_pad|>"
MAX_SEQUENCE_LENGTH = 4096
MAX_CANDIDATE_COUNT = 64
MAX_CANDIDATE_CHARS = 4096
MAX_STATE_CHARS = 65536
MAX_QUESTION_CHARS = 8192
MAX_QUERY_NAME_CHARS = 128
MAX_PROMPT_CHARS = 262144
_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>]*\|>|\[MASK\]")


def load_tokenizer_checked(path: Union[str, Path]):
    """Load a tokenizer and turn serialization/version failures into a clear runtime error."""
    tokenizer_path = Path(path).expanduser()
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception as exc:
        raise RuntimeError(
            f"Tokenizer artifact '{tokenizer_path}' cannot be loaded by the installed "
            "transformers version. Regenerate the tokenizer and checkpoint with the "
            "supported dependency range before deployment."
        ) from exc


def save_tokenizer_checked(
    tokenizer,
    output_dir: Union[str, Path],
    expected_sha256: Optional[str] = None,
):
    """Save a tokenizer, reload it locally, and verify its provenance hash."""
    from src.provenance import sha256_tokenizer

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    expected = expected_sha256 or sha256_tokenizer(tokenizer)
    tokenizer.save_pretrained(str(output_path))
    reloaded = load_tokenizer_checked(output_path)
    actual = sha256_tokenizer(reloaded)
    if actual.casefold() != str(expected).casefold():
        raise RuntimeError(
            "Tokenizer save/reload provenance mismatch: "
            f"before={expected}, after={actual}. "
            "Use the supported transformers version and retrain the checkpoint."
        )
    return reloaded


def normalize_candidates(options: List[str]) -> List[str]:
    """Validate and canonicalize candidate text before model encoding."""
    if not isinstance(options, list):
        raise ValueError("options must be a list of candidate strings")
    if not 2 <= len(options) <= MAX_CANDIDATE_COUNT:
        raise ValueError(
            f"options must contain between 2 and {MAX_CANDIDATE_COUNT} candidates"
        )

    cleaned = []
    for option in options:
        if not isinstance(option, str):
            raise ValueError("each candidate option must be a string")
        raw = option.strip()
        if not raw:
            raise ValueError("candidate options must be non-empty")
        if len(raw) > MAX_CANDIDATE_CHARS:
            raise ValueError(
                f"candidate options must be at most {MAX_CANDIDATE_CHARS} characters"
            )
        if _CONTROL_TOKEN_RE.search(raw):
            raise ValueError("candidate options must not contain control or special tokens")
        cleaned.append(sanitize_text(raw))

    if any(not option for option in cleaned):
        raise ValueError("candidate options must remain non-empty after sanitization")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("candidate options must be distinct after sanitization")
    return cleaned


def _validate_max_length(max_length: int) -> int:
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError("max_length must be an integer")
    if not 1 <= max_length <= MAX_SEQUENCE_LENGTH:
        raise ValueError(f"max_length must be between 1 and {MAX_SEQUENCE_LENGTH}")
    return max_length


def normalize_query_name(name: str, marker: str = DEFAULT_MARKER_TOKEN) -> str:
    """Validate query names before embedding them into a multi-query prompt."""
    if not isinstance(name, str):
        raise ValueError("query names must be strings")
    if not name or name != name.strip():
        raise ValueError("query names must be non-empty and must not have surrounding whitespace")
    if len(name) > MAX_QUERY_NAME_CHARS:
        raise ValueError(f"query names must be at most {MAX_QUERY_NAME_CHARS} characters")
    if (
        _CONTROL_TOKEN_RE.search(name)
        or (marker and marker in name)
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError("query names must not contain control or special tokens")
    return name


def sanitize_text(text: str, marker: str = DEFAULT_MARKER_TOKEN) -> str:
    """
    Sanitize raw user text to prevent special token injection attacks
    that could corrupt marker coordinate alignment.
    """
    if not text:
        return ""
    # Strip any accidental or malicious special-token syntax.
    cleaned = _CONTROL_TOKEN_RE.sub(" ", str(text))
    if marker != DEFAULT_MARKER_TOKEN:
        cleaned = cleaned.replace(marker, " ")
    return cleaned.strip()


def format_decision_prompt(
    state: str,
    question: str,
    options: List[str],
    marker: str = DEFAULT_MARKER_TOKEN,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 2048,
) -> str:
    """
    Format state, instructions, and candidate options into a single prompt string.
    If tokenizer is provided, applies smart Left-Truncation to the state:
    the candidate options and question instructions are ALWAYS preserved,
    while older state tokens are safely truncated from the left.
    """
    max_length = _validate_max_length(max_length)
    if not isinstance(state, str) or len(state) > MAX_STATE_CHARS:
        raise ValueError(f"state must be a string of at most {MAX_STATE_CHARS} characters")
    if not isinstance(question, str) or len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question must be a string of at most {MAX_QUESTION_CHARS} characters")
    options_clean = normalize_candidates(options)
    state_clean = sanitize_text(state, marker=marker)
    q_clean = sanitize_text(question, marker=marker)
    
    options_formatted = []
    for opt_clean in options_clean:
        opt_clean = sanitize_text(opt_clean, marker=marker)
        options_formatted.append(f"{marker} {opt_clean}")
    options_str = "\n".join(options_formatted)
    if len(state_clean) + len(q_clean) + len(options_str) > MAX_PROMPT_CHARS:
        raise ValueError(f"formatted prompt exceeds {MAX_PROMPT_CHARS} characters")
    
    head_str = f"\n[决策指令]: {q_clean}\n[候选动作]:\n{options_str}"
    
    if tokenizer is not None:
        head_tokens = tokenizer.encode(head_str, add_special_tokens=False)
        room = max_length - len(head_tokens) - 16
        if room < 16:
            raise ValueError(
                f"Candidate options ({len(options_clean)} choices, {len(head_tokens)} tokens) "
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
    max_length: int = 2048,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes batch of prompts, locates marker token positions,
    and returns tensors ready for S1DecisionModel.
    Guarantees that candidate markers are NEVER truncated away.
    """
    max_length = _validate_max_length(max_length)
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
    options_per_sample = [normalize_candidates(options) for options in options_per_sample]
    if len(options_per_sample) != len(batch_prompts):
        raise ValueError("options_per_sample must have one candidate list per prompt")
    if not batch_prompts or any(not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS for prompt in batch_prompts):
        raise ValueError(f"batch prompts must be non-empty strings of at most {MAX_PROMPT_CHARS} characters")

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
        token_positions = ((input_ids[b] == marker_token_id) & (attention_mask[b] == 1)).nonzero(as_tuple=True)[0]
        
        if len(token_positions) != num_options:
            raise ValueError(
                f"Sample {b}: Expected exactly {num_options} candidate markers for options, "
                f"but found {len(token_positions)}. This indicates options were truncated, corrupted, "
                f"or an extra marker was injected. Max length is {max_length}."
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

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
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
    max_length: int = 2048,
) -> str:
    """
    Format state and multiple heterogeneous decision queries into a single unified prompt.
    Protects all query instructions and candidate option markers with intelligent left-truncation.
    """
    max_length = _validate_max_length(max_length)
    if not isinstance(state, str) or len(state) > MAX_STATE_CHARS:
        raise ValueError(f"state must be a string of at most {MAX_STATE_CHARS} characters")
    if not isinstance(queries, dict) or not 1 <= len(queries) <= MAX_CANDIDATE_COUNT:
        raise ValueError(f"queries must contain between 1 and {MAX_CANDIDATE_COUNT} entries")
    state_clean = sanitize_text(state, marker=marker)
    
    sections = _build_multi_query_sections(queries, marker)
    total_chars = len(state_clean)
    for q_name, (q_text, opts), section in zip(queries.keys(), queries.values(), sections):
        q_clean = sanitize_text(q_text, marker=marker)
        opts_str = "\n".join(
            f"{marker} {sanitize_text(opt, marker=marker)}" for opt in normalize_candidates(opts)
        )
        total_chars += len(q_name) + len(q_clean) + len(opts_str)
        if total_chars > MAX_PROMPT_CHARS:
            raise ValueError(f"formatted multi-query prompt exceeds {MAX_PROMPT_CHARS} characters")
        
    queries_str = "\n".join(sections)
    if len(state_clean) + len(queries_str) > MAX_PROMPT_CHARS:
        raise ValueError(f"formatted multi-query prompt exceeds {MAX_PROMPT_CHARS} characters")
    
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


def _build_multi_query_sections(
    queries: Dict[str, Tuple[str, List[str]]],
    marker: str,
) -> List[str]:
    """Build validated query sections exactly as they appear in the prompt."""
    sections = []
    for q_name, (q_text, opts) in queries.items():
        normalize_query_name(q_name, marker=marker)
        if not isinstance(q_text, str) or len(q_text) > MAX_QUESTION_CHARS:
            raise ValueError(f"query text must be a string of at most {MAX_QUESTION_CHARS} characters")
        q_clean = sanitize_text(q_text, marker=marker)
        opts_formatted = [
            f"{marker} {sanitize_text(opt, marker=marker)}" for opt in normalize_candidates(opts)
        ]
        opts_str = "\n".join(opts_formatted)
        sections.append(f"\n[决策指令: {q_name}]: {q_clean}\n[候选动作: {q_name}]:\n{opts_str}")
    return sections


def encode_multi_query_batch(
    tokenizer: PreTrainedTokenizer,
    states: List[str],
    queries_per_sample: List[Dict[str, Tuple[str, List[str]]]],
    marker: str = DEFAULT_MARKER_TOKEN,
    max_length: int = 2048,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Tokenizes a batch of multi-query prompts and locates marker token coordinates
    for each query in the prompt, enabling single-forward-pass parallel evaluation.
    """
    max_length = _validate_max_length(max_length)
    marker_token_id = tokenizer.convert_tokens_to_ids(marker)
    if marker_token_id is None or marker_token_id == tokenizer.unk_token_id:
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            marker_token_id = tokenizer.mask_token_id
        else:
            marker_token_id = tokenizer.eos_token_id

    if len(states) != len(queries_per_sample):
        raise ValueError("states and queries_per_sample must have the same length")
    if not states:
        raise ValueError("states and queries_per_sample must be non-empty")
    for queries in queries_per_sample:
        if not isinstance(queries, dict) or not queries:
            raise ValueError("each query sample must contain a non-empty query mapping")
        names = [normalize_query_name(name, marker=marker) for name in queries]
        if len(set(names)) != len(names):
            raise ValueError("query names must be unique")
    query_names = list(queries_per_sample[0])
    if not query_names or any(list(queries) != query_names for queries in queries_per_sample):
        raise ValueError("all query samples must contain the same query names in the same order")
    batch_size = len(states)
    batch_prompts = []
    section_end_chars = []
    for b in range(batch_size):
        sections = _build_multi_query_sections(queries_per_sample[b], marker)
        queries_str = "\n".join(sections)
        p = format_multi_query_prompt(
            state=states[b],
            queries=queries_per_sample[b],
            marker=marker,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        batch_prompts.append(p)
        if not p.endswith(queries_str):
            raise ValueError("multi-query prompt construction lost its query sections")
        query_base = len(p) - len(queries_str)
        cursor = query_base
        boundaries = []
        for index, section in enumerate(sections):
            section_end = cursor + len(section)
            boundaries.append(section_end + 1 if index < len(sections) - 1 else len(p))
            cursor = section_end + (1 if index < len(sections) - 1 else 0)
        section_end_chars.append(boundaries)

    try:
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping", None)
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offset_mapping = None
    encoded = encoded.to(device)
    if offset_mapping is not None:
        offset_mapping = offset_mapping.to(device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    seq_len = input_ids.shape[1]

    # Query names from sample 0
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
        token_positions = ((input_ids[b] == marker_token_id) & (attention_mask[b] == 1)).nonzero(as_tuple=True)[0]
        total_expected_markers = sum(len(opts) for _, opts in queries_per_sample[b].values())
        if len(token_positions) != total_expected_markers:
            raise ValueError(
                f"Sample {b}: Expected exactly {total_expected_markers} markers across {len(query_names)} queries, "
                f"but found {len(token_positions)}. Tokens were truncated, corrupted, or an extra marker was injected."
            )
        
        non_pad = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
        seq_end = non_pad[-1].item() + 1 if len(non_pad) > 0 else seq_len

        offset = 0
        for query_index, (q_name, (q_text, opts)) in enumerate(queries_per_sample[b].items()):
            num_opts = len(opts)
            for k in range(num_opts):
                global_idx = offset + k
                s_pos = token_positions[global_idx].item()
                if global_idx < total_expected_markers - 1:
                    e_pos = token_positions[global_idx + 1].item()
                else:
                    e_pos = seq_end

                if k == num_opts - 1 and query_index < len(section_end_chars[b]) - 1:
                    boundary = section_end_chars[b][query_index]
                    if offset_mapping is not None:
                        e_pos = seq_end
                        for token_index, (start, end) in enumerate(offset_mapping[b].tolist()):
                            if token_index >= seq_end:
                                break
                            if end > start and end > boundary:
                                e_pos = token_index
                                break
                    else:
                        next_marker = token_positions[global_idx + 1].item()
                        next_query_name = query_names[query_index + 1]
                        header = (
                            f"\n[决策指令: {next_query_name}]: "
                            f"{sanitize_text(queries_per_sample[b][next_query_name][0], marker=marker)}\n"
                            f"[候选动作: {next_query_name}]:\n"
                        )
                        header_ids = tokenizer.encode(header, add_special_tokens=False)
                        if header_ids and next_marker >= len(header_ids):
                            candidate_start = next_marker - len(header_ids)
                            if input_ids[b, candidate_start:next_marker].tolist() == header_ids:
                                e_pos = candidate_start

                opt_start = s_pos + 1 if e_pos > s_pos + 1 else s_pos
                query_markers[q_name]["indices"][b, k] = s_pos
                query_markers[q_name]["mask"][b, k] = True
                query_markers[q_name]["spans"][b, k, 0] = opt_start
                query_markers[q_name]["spans"][b, k, 1] = e_pos
            offset += num_opts

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_markers": query_markers,
    }
