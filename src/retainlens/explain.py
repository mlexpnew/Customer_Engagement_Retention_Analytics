"""Why is this customer at risk? Reason codes a relationship manager can act on.

Model-agnostic: for each driver, reset it to a typical "healthy" value and measure how
much the churn probability drops. The biggest drops are the reasons. Each reason maps to
a concrete retention action, so the output is a call script, not just a score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import add_features

# driver -> (columns to reset, label, suggested action)
DRIVERS = {
    "inactive": (["IsActiveMember"], "Inactive in the last period",
                 "Re-engage: personal call, app onboarding, activity-linked reward"),
    "products": (["NumOfProducts"], "Product holding is a churn signal",
                 "Review product fit: 1 product = cross-sell, 3+ = simplify / consolidate"),
    "age": (["Age"], "Life-stage segment with high churn",
            "Offer life-stage products: pension, wealth review, family banking"),
    "balance": (["Balance"], "Balance pattern linked to leaving",
                "Wealth check-in; savings rate or term deposit offer"),
    "geography": (["Geography"], "Market with elevated churn",
                  "Apply market-specific retention offer"),
    "credit": (["CreditScore"], "Credit score band with higher churn",
               "Proactive credit health review and limit offer"),
    "tenure": (["Tenure"], "Tenure stage with higher churn",
               "Anniversary benefit or loyalty upgrade"),
}  # fmt: skip


def reference_profile(train_df: pd.DataFrame) -> dict:
    """Typical values among customers who stayed: the 'healthy' baseline."""
    stayed = train_df[train_df["Exited"] == 0]
    return {
        "IsActiveMember": 1,
        "NumOfProducts": 2,
        "Age": float(stayed["Age"].median()),
        "Balance": float(stayed["Balance"].median()),
        "Geography": stayed["Geography"].mode().iloc[0],
        "CreditScore": float(stayed["CreditScore"].median()),
        "Tenure": float(stayed["Tenure"].median()),
    }


def reasons(bundle, customers: pd.DataFrame, reference: dict, top_k: int = 3) -> list[list[dict]]:
    base = bundle.predict_proba(customers)
    deltas = {}
    for key, (cols, _, _) in DRIVERS.items():
        alt = customers.copy()
        for c in cols:
            alt[c] = reference[c]
        alt = add_features(alt)
        deltas[key] = base - bundle.predict_proba(alt)
    out = []
    for i in range(len(customers)):
        ranked = sorted(((deltas[k][i], k) for k in DRIVERS), reverse=True)
        out.append([
            {"driver": k, "reason": DRIVERS[k][1], "action": DRIVERS[k][2],
             "risk_reduction": round(float(d), 3)}
            for d, k in ranked[:top_k] if d > 0.01
        ])  # fmt: skip
    return out


def global_importance(bundle, df: pd.DataFrame, reference: dict, sample: int = 1500) -> dict:
    """Mean risk reduction per driver across customers: what drives churn overall."""
    s = df.sample(min(sample, len(df)), random_state=0)
    base = bundle.predict_proba(s)
    out = {}
    for key, (cols, label, _) in DRIVERS.items():
        alt = s.copy()
        for c in cols:
            alt[c] = reference[c]
        out[label] = round(float(np.mean(base - bundle.predict_proba(add_features(alt)))), 4)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
