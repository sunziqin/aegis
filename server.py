# -*- coding: utf-8 -*-
"""
Aegis-S1 Production Inference Microservice (FastAPI).
Provides sub-30ms decision API with Conformal Risk Guarantees.
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
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_s1 import AegisRouter

app = FastAPI(
    title="Aegis-S1: Provably Safe System-1 Decision Service",
    description="Non-autoregressive sub-30ms decision reflexes for AI Agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router_instance: Optional[AegisRouter] = None


class DecisionRequest(BaseModel):
    state: str = Field(..., description="Current system/user state or context")
    question: str = Field(..., description="Decision instructions or intent prompt")
    candidates: List[str] = Field(..., description="List of candidate option strings")
    alpha: float = Field(0.05, description="Conformal error tolerance (default 0.05 for 95% safety guarantee)")


class DecisionResponse(BaseModel):
    ok: bool = True
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
    print("[*] Aegis-S1 Router successfully loaded and ready for sub-30ms requests!")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "aegis-s1",
        "version": "1.0.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/api/v1/decide", response_model=DecisionResponse)
def decide(req: DecisionRequest):
    if router_instance is None:
        raise HTTPException(status_code=503, detail="Model router is still loading...")
        
    t0 = time.perf_counter()
    res = router_instance.decide(
        state=req.state,
        question=req.question,
        candidates=req.candidates,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return DecisionResponse(
        ok=True,
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
