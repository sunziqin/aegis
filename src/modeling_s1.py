# -*- coding: utf-8 -*-
"""
S1-Decision: Non-autoregressive System 1 Decision Model
Featuring Dynamic Option-Marker Scorer, Inter-Option Attention, and Escalate Gate.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class InterOptionTransformer(nn.Module):
    """
    Lightweight 2-layer Transformer encoder to model mutual competition
    and semantic interaction among variable-length candidate options.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, ff_dim: Optional[int] = None, num_layers: int = 2):
        super().__init__()
        if ff_dim is None:
            ff_dim = hidden_dim * 2
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, option_embeds: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            option_embeds: [Batch, K, HiddenDim]
            option_mask: [Batch, K] (True for valid option, False for padded)
        Returns:
            contextualized_embeds: [Batch, K, HiddenDim]
        """
        # nn.TransformerEncoder expects key_padding_mask=True for positions to be IGNORED
        key_padding_mask = ~option_mask
        return self.encoder(option_embeds, src_key_padding_mask=key_padding_mask)


class DynamicOptionMarkerHead(nn.Module):
    """
    Decision head that extracts marker tokens (e.g. [MASK] or <opt>)
    for dynamically supplied candidate options, runs inter-option attention,
    and produces calibrated choice probabilities and an escalate risk score.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_inter_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Inter-option interaction
        self.inter_option_attn = InterOptionTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_inter_layers,
        )
        
        # Linear scoring projection
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Act / Escalate Risk Gate (estimates whether input is ambiguous or out-of-domain)
        self.escalate_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence_hidden_states: torch.Tensor,
        marker_indices: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_hidden_states: [Batch, SeqLen, HiddenDim]
            marker_indices: [Batch, MaxK] (token indices of candidate markers)
            marker_mask: [Batch, MaxK] (bool, True for real options, False for padding)
        Returns:
            logits: [Batch, MaxK]
            probs: [Batch, MaxK]
            escalate_score: [Batch]
        """
        batch_size, max_k = marker_indices.shape
        head_dtype = self.scorer[1].weight.dtype
        sequence_hidden_states = sequence_hidden_states.to(dtype=head_dtype)
        hidden_dim = sequence_hidden_states.shape[-1]
        
        # 1. Gather marker representations: [Batch, MaxK, HiddenDim]
        expanded_indices = marker_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        safe_indices = torch.clamp(expanded_indices, min=0)
        option_embeds = torch.gather(sequence_hidden_states, dim=1, index=safe_indices)
        
        # Zero out padding representations
        option_embeds = option_embeds * marker_mask.unsqueeze(-1).to(dtype=sequence_hidden_states.dtype)
        
        # 2. Inter-Option Cross-Attention
        contextual_options = self.inter_option_attn(option_embeds, marker_mask)
        
        # 3. Calculate Option Logits
        raw_logits = self.scorer(contextual_options).squeeze(-1)  # [Batch, MaxK]
        
        # Mask out padding options with large negative value for softmax
        masked_logits = raw_logits.masked_fill(~marker_mask, -1e4 if head_dtype == torch.bfloat16 else -1e9)
        probs = F.softmax(masked_logits, dim=-1)
        probs = probs * marker_mask.to(dtype=probs.dtype)  # Ensure padded options have 0 probability
        
        # 4. Global Escalate Risk Gate
        cls_hidden = sequence_hidden_states[:, 0, :]
        escalate_score = self.escalate_gate(cls_hidden).squeeze(-1)  # [Batch]
        
        return masked_logits, probs, escalate_score


class S1DecisionModel(nn.Module):
    """
    Complete S1 Model integrating a Transformer Encoder backbone (ModernBERT / Qwen)
    with the DynamicOptionMarkerHead.
    """
    def __init__(self, backbone: nn.Module, hidden_dim: int, num_heads: int = 8, num_inter_layers: int = 2):
        super().__init__()
        self.backbone = backbone
        self.decision_head = DynamicOptionMarkerHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inter_layers=num_inter_layers,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_heads: int = 8,
        num_inter_layers: int = 2,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=False)
        hidden_dim = getattr(config, "hidden_size", None) or getattr(config, "d_model", 1024)
        
        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch_dtype or torch.bfloat16,
        )
        model = cls(
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inter_layers=num_inter_layers,
        )
        if torch_dtype is not None:
            model.decision_head.to(dtype=torch_dtype)
        elif hasattr(backbone, "dtype"):
            model.decision_head.to(dtype=backbone.dtype)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_indices: torch.Tensor,
        marker_mask: torch.Tensor,
        bidirectional: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Single non-autoregressive forward pass returning decision predictions.
        If bidirectional=True, transforms 2D causal mask into 4D full bidirectional
        mask, turning Decoder LLM into a modern Bidirectional Decision Encoder.
        """
        if bidirectional and attention_mask.dim() == 2:
            batch_size, seq_len = attention_mask.shape
            mask_dtype = torch.bfloat16 if input_ids.is_cuda else torch.float32
            mask_4d = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=mask_dtype, device=input_ids.device)
            for b in range(batch_size):
                pad_idx = (attention_mask[b] == 0).nonzero(as_tuple=True)[0]
                if len(pad_idx) > 0:
                    mask_4d[b, 0, :, pad_idx] = -1e4
                    mask_4d[b, 0, pad_idx, :] = -1e4
            attn_mask_to_pass = mask_4d
        else:
            attn_mask_to_pass = attention_mask

        outputs = self.backbone(input_ids=input_ids, attention_mask=attn_mask_to_pass)
        sequence_hidden = outputs.last_hidden_state  # [Batch, SeqLen, HiddenDim]
        
        logits, probs, escalate = self.decision_head(
            sequence_hidden_states=sequence_hidden,
            marker_indices=marker_indices,
            marker_mask=marker_mask,
        )
        
        confidence, best_idx = torch.max(probs, dim=-1)
        
        return {
            "logits": logits,
            "probs": probs,
            "confidence": confidence,
            "best_choice_idx": best_idx,
            "escalate_risk": escalate,
        }
