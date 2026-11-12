"""Customer data: schema, validation and features.

Protected attributes (Gender) are kept for fairness audits but never used as model
inputs: dropping them costs about 0.002 ROC-AUC on this data (see README), which is a
small price for retention offers that do not depend on a customer's gender.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
           "IsActiveMember", "EstimatedSalary"]  # fmt: skip
CATEGORICAL = ["Geography"]
PROTECTED = ["Gender"]
ID, TARGET = "CustomerId", "Exited"
ENGINEERED = ["ZeroBalance", "BalanceToSalary", "ProductsPerTenureYear", "InactiveMultiProduct"]
FEATURES = NUMERIC + ENGINEERED + CATEGORICAL

RANGES = {  # plausible bounds; rows outside are rejected with a reason, not silently scored
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 50),
    "Balance": (0, 1e8),
    "NumOfProducts": (1, 10),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0, 1e8),
}


@dataclass
class Validation:
    valid: pd.DataFrame
    rejected: pd.DataFrame  # original rows plus a "reason" column


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> Validation:
    missing = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    reasons = pd.Series("", index=df.index, dtype="object")
    for col, (lo, hi) in RANGES.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.isna() | (vals < lo) | (vals > hi)
        reasons[bad] += f"{col} outside {lo}-{hi:g}; "
    geo_bad = df["Geography"].isna() | (df["Geography"].astype(str).str.strip() == "")
    reasons[geo_bad] += "Geography missing; "
    ok = reasons == ""
    rejected = df[~ok].copy()
    rejected["reason"] = reasons[~ok].str.rstrip("; ")
    return Validation(valid=df[ok].copy(), rejected=rejected)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col])
    out["Geography"] = out["Geography"].astype(str).str.strip().str.title()
    out["ZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["BalanceToSalary"] = out["Balance"] / out["EstimatedSalary"].clip(lower=1000)
    out["ProductsPerTenureYear"] = out["NumOfProducts"] / (out["Tenure"] + 1)
    inactive_multi = (out["IsActiveMember"] == 0) & (out["NumOfProducts"] >= 2)
    out["InactiveMultiProduct"] = inactive_multi.astype(int)
    return out


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified holdout; the test set is touched once, for the final report."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, g in df.groupby(TARGET):
        n = round(len(g) * test_size)
        test_idx.extend(rng.choice(g.index.to_numpy(), size=n, replace=False))
    test = df.loc[sorted(test_idx)]
    return df.drop(index=test.index), test
