"""HTTP API for CRM and core-banking integrations.

    uvicorn retainlens.api:app --port 8000

Set RETAINLENS_API_KEYS (comma separated) to require an X-API-Key header.
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .plan import Assumptions
from .service import drift, load_model, plan, score

MAX_ROWS = 50_000

app = FastAPI(
    title="RetainLens",
    version=__version__,
    description="Churn early warning and retention planning for retail banks.",
)


class Customer(BaseModel):
    CustomerId: int | str
    CreditScore: float
    Geography: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float
    Gender: str | None = None  # accepted for reporting, never used by the model


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_ROWS)


class PlanRequest(Batch):
    budget: float = Field(gt=0)
    contact_cost: float = Field(40.0, gt=0)
    success_rate: float = Field(0.30, gt=0, le=1)


def auth(x_api_key: str = Header(default="")) -> None:
    keys = {k.strip() for k in os.getenv("RETAINLENS_API_KEYS", "").split(",") if k.strip()}
    if keys and x_api_key not in keys:
        raise HTTPException(401, "missing or invalid X-API-Key")


def _df(batch: Batch) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in batch.customers])


@app.get("/health")
def health() -> dict:
    m = load_model()
    return {"status": "ok", "version": __version__, "model": m.name,
            "trained_at": m.metadata.get("trained_at"),
            "holdout_roc_auc": m.metadata.get("holdout", {}).get("roc_auc")}  # fmt: skip


@app.post("/v1/score", dependencies=[Depends(auth)])
def score_endpoint(batch: Batch) -> dict:
    scored, rejected = score(load_model(), _df(batch))
    return {
        "scores": scored.to_dict(orient="records"),
        "rejected": rejected.to_dict(orient="records"),
    }


@app.post("/v1/plan", dependencies=[Depends(auth)])
def plan_endpoint(req: PlanRequest) -> dict:
    a = Assumptions(contact_cost=req.contact_cost, success_rate=req.success_rate)
    out = plan(load_model(), _df(req), req.budget, a)
    return {
        "contacts": out.to_dict(orient="records"),
        "spend": round(len(out) * a.contact_cost, 2),
        "expected_net_value": float(out["expected_net_value"].sum()) if len(out) else 0.0,
    }


@app.post("/v1/drift", dependencies=[Depends(auth)])
def drift_endpoint(batch: Batch) -> dict:
    return drift(load_model(), _df(batch))
