#!/usr/bin/env python3
"""v9.7 Production — FastAPI endpoint for ReTop models.

Usage:
    python -m hmn.api
    # or
    uvicorn hmn.api:app --host 0.0.0.0 --port 8000

Docker:
    docker build -t retop .
    docker run -p 8000:8000 retop
"""

import os
import sys
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer

from hmn import HMNConfig, create_model
from hmn.recipe import resolve_device

app = FastAPI(title="ReTop API", version="0.9.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
tokenizer = None
device = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    mode: str = "hard"


class GenerateResponse(BaseModel):
    text: str
    tokens: list[int]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
def load_model():
    global model, tokenizer, device

    checkpoint = os.getenv("RETOP_CHECKPOINT", "hmn_v33.pt")
    tokenizer = Tokenizer.from_file("retop_tokenizer.json")
    device = resolve_device(None)

    cfg = HMNConfig(
        vocab_size=tokenizer.get_vocab_size(),
        dim=96, n_layers=3, variant="ssm",
        seam_addr=True, stem_addr=True,
        asi_id=tokenizer.token_to_id("<|assistant|>"),
    )
    model = create_model(cfg).to(device)

    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    model.eval()


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from hmn.recipe import decode_v33

    t0 = time.time()
    tokens = tokenizer.encode(req.prompt).ids
    x = torch.tensor([tokens], device=device)

    txt, gate, _ = decode_v33(
        model, tokenizer, x,
        max_new=req.max_new_tokens,
        temp=req.temperature,
        top_k=req.top_k,
        mode=req.mode,
        seam=True,
        device=device,
    )

    out_tokens = tokenizer.encode(txt).ids
    elapsed = (time.time() - t0) * 1000

    return GenerateResponse(text=txt, tokens=out_tokens, elapsed_ms=elapsed)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
