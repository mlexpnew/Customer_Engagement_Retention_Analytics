"""Scoring service shared by the API, CLI and app: one code path, one set of rules."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import ID, add_features, validate
from .explain import reasons
from .model import ModelBundle
from .monitor import drift_report
from .plan import Assumptions, expected_net, select


def default_model_path() -> str:
    """CHURNSHIELD_MODEL, else models/ in the working directory, else next to the source tree."""
    if os.getenv("CHURNSHIELD_MODEL"):
        return os.environ["CHURNSHIELD_MODEL"]
    local = Path("models/churnshield.joblib")
    if local.exists():
        return str(local)
    return str(Path(__file__).resolve().parents[2] / "models" / "churnshield.joblib")


@lru_cache(maxsize=2)
def load_model(path: str | None = None) -> ModelBundle:
    return ModelBundle.load(path or default_model_path())


def score(
    bundle: ModelBundle, customers: pd.DataFrame, top_reasons: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scored rows (probability, band, reasons) and rejected rows (with why)."""
    v = validate(customers)
    if v.valid.empty:
        return pd.DataFrame(), v.rejected
    df = add_features(v.valid)
    p = bundle.predict_proba(df)
    out = pd.DataFrame({
        ID: df[ID].to_numpy() if ID in df else range(len(df)),
        "churn_probability": p.round(4),
        "risk_band": bundle.band(p),
    }, index=df.index)  # fmt: skip
    out["reasons"] = reasons(bundle, df, bundle.metadata["reference_profile"], top_reasons)
    return out, v.rejected


def plan(bundle: ModelBundle, customers: pd.DataFrame, budget: float,
         a: Assumptions = Assumptions()) -> pd.DataFrame:  # fmt: skip
    """Contact list within budget, best expected value first, with talking points."""
    scored, _ = score(bundle, customers)
    if scored.empty:
        return scored
    df = add_features(validate(customers).valid)
    p = scored["churn_probability"].to_numpy()
    ev = expected_net(p, df, a)
    chosen = select(p, df, budget, a)
    out = scored.iloc[chosen].copy()
    out["customer_value"] = a.value(df.iloc[chosen]).round(0)
    out["expected_net_value"] = ev[chosen].round(0)
    out["rank"] = range(1, len(out) + 1)
    return out


def drift(bundle: ModelBundle, batch: pd.DataFrame) -> dict:
    ref = getattr(bundle, "reference", None)
    if ref is None:
        raise ValueError("model has no reference sample; retrain with churnshield.train")
    return drift_report(ref, add_features(validate(batch).valid))
