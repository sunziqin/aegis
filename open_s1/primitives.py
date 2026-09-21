# -*- coding: utf-8 -*-
"""
Aegis-S1 Type-Safe Decision Primitives.
Matches and extends TypeSafe Jev & Laya decision primitives:
- Choice: Categorical selection over dynamic runtime candidates with Conformal Guarantees.
- Score: Ordinal / continuous evaluation via discrete expectation E[S] and variance Var[S].
- Noul (Boolean): Calibrated binary decision predicate (Yes/No, Pass/Fail, Escalation check).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


@dataclass
class Choice:
    """
    Categorical decision primitive over dynamic string candidates.
    Args:
        options: List of candidate actions or categories.
        question: Description or instruction of what to decide.
    """
    options: List[str]
    question: str = "请选择最匹配的候选动作"

    def __post_init__(self):
        if not self.options or len(self.options) < 2:
            raise ValueError("Choice primitive requires at least 2 candidate options.")


@dataclass
class Score:
    """
    Ordinal / continuous scoring primitive.
    Discretizes the scale into ordinal bins and computes the expectation E[S] = sum(v_k * p_k).
    Args:
        min_val: Minimum score (e.g. 1.0)
        max_val: Maximum score (e.g. 5.0 or 100.0)
        steps: Number of discrete steps/bins (default: 5)
        labels: Optional human-readable labels for each step (e.g. ['极低', '低', '中', '高', '紧急'])
        question: Description or instruction of what to score.
    """
    min_val: float = 1.0
    max_val: float = 5.0
    steps: int = 5
    labels: Optional[List[str]] = None
    question: str = "请对该指标进行量化评估"

    def __post_init__(self):
        if self.steps < 2:
            raise ValueError("Score primitive requires at least 2 steps.")
        if self.min_val >= self.max_val:
            raise ValueError("min_val must be strictly less than max_val.")
        
        self.values = [
            self.min_val + i * (self.max_val - self.min_val) / (self.steps - 1)
            for i in range(self.steps)
        ]
        
        if self.labels is None:
            self.labels = [f"{v:.1f}分" for v in self.values]
        elif len(self.labels) != self.steps:
            raise ValueError(f"labels count ({len(self.labels)}) must match steps ({self.steps}).")

    @property
    def options(self) -> List[str]:
        return [f"{v:.1f}: {lbl}" for v, lbl in zip(self.values, self.labels)]


@dataclass
class Noul:
    """
    Binary decision predicate primitive (named after Jev's Noul primitive).
    Also aliased as Boolean.
    Args:
        threshold: Decision threshold for positive outcome (default: 0.5)
        question: Clarifying question for the boolean decision (e.g. '是否需要转人工？')
        true_label: Semantic label for True (default: '是/通过')
        false_label: Semantic label for False (default: '否/拒绝')
    """
    threshold: float = 0.5
    question: str = "判定该条件是否成立"
    true_label: str = "成立/是/同意"
    false_label: str = "不成立/否/拒绝"

    @property
    def options(self) -> List[str]:
        # [False, True]
        return [self.false_label, self.true_label]


# Alias for intuitive typing
Boolean = Noul


# =========================================================================
# Result Dataclasses
# =========================================================================

@dataclass
class ChoiceResult:
    selected_option: str
    selected_index: int
    confidence: float
    verdict: str  # "act" or "escalate"
    prediction_set: List[str]
    prediction_set_size: int
    probabilities: Dict[str, float]
    explanation: str

    def is_act(self) -> bool:
        return self.verdict == "act"


@dataclass
class ScoreResult:
    score: float
    std: float
    confidence: float
    verdict: str  # "act" or "escalate" (if uncertainty/std is too high)
    probabilities: Dict[str, float]
    explanation: str

    def is_act(self) -> bool:
        return self.verdict == "act"


@dataclass
class NoulResult:
    value: bool
    probability: float
    confidence: float
    verdict: str  # "act" or "escalate"
    probabilities: Dict[str, float]
    explanation: str

    def is_act(self) -> bool:
        return self.verdict == "act"

    def __bool__(self) -> bool:
        return self.value


@dataclass
class EvaluationResult:
    """
    Container for parallel multi-query evaluation results in a single forward pass.
    Allows both dictionary-style access (res['intent']) and attribute access (res.intent).
    """
    results: Dict[str, Union[ChoiceResult, ScoreResult, NoulResult]]
    overall_verdict: str
    escalate_risk: float
    latency_ms: float

    def __getitem__(self, key: str) -> Union[ChoiceResult, ScoreResult, NoulResult]:
        return self.results[key]

    def __getattr__(self, key: str) -> Any:
        if key in self.results:
            return self.results[key]
        raise AttributeError(f"'EvaluationResult' object has no attribute '{key}'")

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    def __repr__(self) -> str:
        res_str = ", ".join(f"{k}={v.__class__.__name__}" for k, v in self.results.items())
        return f"EvaluationResult(verdict='{self.overall_verdict}', risk={self.escalate_risk:.3f}, latency={self.latency_ms:.1f}ms, queries=[{res_str}])"
