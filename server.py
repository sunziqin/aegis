# -*- coding: utf-8 -*-
"""
Aegis-S1 Production Inference Microservice (FastAPI).
Provides a low-latency decision API with conformal risk checks.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_s1 import AegisRouter, __version__
from src.tokenizer_utils import (
    MAX_CANDIDATE_CHARS,
    MAX_CANDIDATE_COUNT,
    MAX_QUESTION_CHARS,
    MAX_STATE_CHARS,
    normalize_candidates,
)

app = FastAPI(
    title="Aegis-S1: Provably Safe System-1 Decision Service",
    description="Non-autoregressive decision reflexes for AI agents; benchmark latency on target hardware.",
    version=__version__,
)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "AEGIS_ALLOWED_ORIGINS", "http://127.0.0.1:18099,http://localhost:18099"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

router_instance: Optional[AegisRouter] = None


class DecisionRequest(BaseModel):
    state: str = Field(..., max_length=MAX_STATE_CHARS, description="Current system/user state or context")
    question: str = Field(..., max_length=MAX_QUESTION_CHARS, description="Decision instructions or intent prompt")
    candidates: List[str] = Field(
        ...,
        min_length=2,
        max_length=MAX_CANDIDATE_COUNT,
        description="Candidate options (minimum 2 distinct)",
    )
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Conformal error tolerance (0 < alpha < 1, default 0.05 for 95% safety guarantee)")

    @field_validator("candidates")
    @classmethod
    def validate_candidates(cls, v: List[str]) -> List[str]:
        cleaned = normalize_candidates(v)
        if any(len(candidate) > MAX_CANDIDATE_CHARS for candidate in cleaned):
            raise ValueError(f"Candidate options must be at most {MAX_CANDIDATE_CHARS} characters")
        return cleaned


class DecisionResponse(BaseModel):
    ok: bool = True
    can_act: bool = Field(..., description="Whether action is cleared for autonomous execution (Tri-Gate pass)")
    selected_option: str
    selected_index: int
    confidence: float
    escalate_risk: float
    conformal_verdict: str  # 'act', 'escalate', or 'reject'
    prediction_set: List[str]
    explanation: str
    probabilities: Dict[str, float]
    latency_ms: float


@app.on_event("startup")
def startup_event():
    global router_instance
    print("[*] Initializing Aegis-S1 Router on GPU...")
    router_instance = AegisRouter.load()
    print("[*] Aegis-S1 Router successfully loaded; benchmark latency on the deployment hardware.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if router_instance is not None else "loading",
        "service": "aegis-s1",
        "version": __version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/api/v1/decide", response_model=DecisionResponse)
def decide(req: DecisionRequest):
    if router_instance is None:
        raise HTTPException(status_code=503, detail="Model router is still loading...")
        
    t0 = time.perf_counter()
    try:
        res = router_instance.decide(
            state=req.state,
            question=req.question,
            candidates=req.candidates,
            alpha=req.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return DecisionResponse(
        ok=True,
        can_act=res["can_act"],
        selected_option=res["selected_option"],
        selected_index=res["selected_index"],
        confidence=res["confidence"],
        escalate_risk=res["escalate_risk"],
        conformal_verdict=res["conformal_verdict"],
        prediction_set=res["prediction_set"],
        explanation=res["explanation"],
        probabilities=res["probabilities"],
        latency_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18099)
