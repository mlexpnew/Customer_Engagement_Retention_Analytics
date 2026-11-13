"""Production monitoring: data drift (PSI) and fairness by segment.

PSI (population stability index) compares this month's feature distribution with the
training data. Industry rule of thumb, used in bank model risk management:
below 0.10 stable, 0.10 to 0.25 watch, above 0.25 investigate / retrain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import CATEGORICAL, NUMERIC

WATCH, ACT = 0.10, 0.25


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    eps = 1e-4
    if expected.dtype == object or expected.nunique() <= 5:
        cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        a = actual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    e, a = e.to_numpy() + eps, a.to_numpy() + eps
    return round(float(np.sum((a - e) * np.log(a / e))), 4)


def drift_report(reference: pd.DataFrame, batch: pd.DataFrame) -> dict:
    rows = {}
    for col in NUMERIC + CATEGORICAL:
        v = psi(reference[col], batch[col])
        rows[col] = {"psi": v, "status": "act" if v > ACT else "watch" if v > WATCH else "ok"}
    worst = max(rows.values(), key=lambda r: r["psi"])
    return {
        "features": rows,
        "overall": worst["status"],
        "retrain_recommended": worst["status"] == "act",
    }


def segment_report(df: pd.DataFrame, p: np.ndarray, flagged: np.ndarray, by: str) -> dict:
    """Per segment: actual churn, model AUC, share flagged, and recall of real churners."""
    out = {}
    y = df["Exited"].to_numpy()
    for seg in sorted(df[by].astype(str).unique()):
        m = (df[by].astype(str) == seg).to_numpy()
        churners = m & (y == 1)
        out[seg] = {
            "customers": int(m.sum()),
            "churn_rate": round(float(y[m].mean()), 3),
            "roc_auc": round(roc_auc_score(y[m], p[m]), 3) if len(set(y[m])) == 2 else None,
            "flagged_rate": round(float(flagged[m].mean()), 3),
            "recall_of_churners": round(float(flagged[churners].mean()), 3)
            if churners.any() else None,
        }  # fmt: skip
    return out
