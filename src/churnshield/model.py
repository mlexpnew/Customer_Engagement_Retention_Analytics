"""Train, compare, calibrate and persist the churn model.

Candidates are compared with 5-fold cross-validation on the training split only. The
winner is calibrated (isotonic), because the retention planner multiplies probabilities
by money: a score of 0.3 has to mean roughly a 30% chance of leaving.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import __version__
from .data import CATEGORICAL, ENGINEERED, FEATURES, NUMERIC, TARGET

SEED = 42


def _pre() -> ColumnTransformer:
    return ColumnTransformer(
        [("num", StandardScaler(), NUMERIC + ENGINEERED),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]  # fmt: skip
    )


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("pre", _pre()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "random_forest": Pipeline(
            [("pre", _pre()),
             ("clf", RandomForestClassifier(400, min_samples_leaf=5, n_jobs=-1, random_state=SEED))]
        ),  # fmt: skip
        "gradient_boosting": Pipeline(
            [("pre", _pre()),
             ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                    max_leaf_nodes=15, l2_regularization=1.0,
                                                    random_state=SEED))]
        ),  # fmt: skip
    }


# Cheapest to serve first: on a near-tie the smaller, faster, easier-to-explain model wins.
SIMPLICITY = ["logistic_regression", "gradient_boosting", "random_forest"]
NEAR_TIE = 0.003  # ROC-AUC points; below typical fold-to-fold noise


def choose(comparison: dict) -> str:
    top = max(v["cv_roc_auc"] for v in comparison.values())
    return next(m for m in SIMPLICITY if top - comparison[m]["cv_roc_auc"] <= NEAR_TIE)


@dataclass
class ModelBundle:
    model: object
    name: str
    threshold_high: float
    threshold_medium: float
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Isotonic steps can hit exactly 0 or 1; no customer is certain to stay or leave.
        return np.clip(self.model.predict_proba(X[FEATURES])[:, 1], 0.01, 0.99)

    def band(self, p: np.ndarray) -> np.ndarray:
        return np.where(p >= self.threshold_high, "high",
                        np.where(p >= self.threshold_medium, "medium", "low"))  # fmt: skip

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2))

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        return joblib.load(path)


def data_fingerprint(df: pd.DataFrame) -> str:
    digest = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(digest).hexdigest()[:12]


def train(train_df: pd.DataFrame, folds: int = 5) -> tuple[ModelBundle, dict]:
    X, y = train_df[FEATURES], train_df[TARGET]
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    comparison = {}
    for name, pipe in candidates().items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        comparison[name] = {"cv_roc_auc": round(scores.mean(), 4), "cv_std": round(scores.std(), 4)}
    best = choose(comparison)
    calibrated = CalibratedClassifierCV(candidates()[best], method="isotonic", cv=cv)
    calibrated.fit(X, y)
    # Risk bands from the training score distribution: top 10% high, next 20% medium
    p = calibrated.predict_proba(X)[:, 1]
    hi, med = float(np.quantile(p, 0.90)), float(np.quantile(p, 0.70))
    meta = {
        "version": __version__,
        "model": best,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_rows": len(train_df),
        "train_fingerprint": data_fingerprint(train_df[FEATURES + [TARGET]]),
        "features": FEATURES,
        "excluded_protected": ["Gender"],
        "cv_comparison": comparison,
        "risk_bands": {"high": round(hi, 4), "medium": round(med, 4)},
    }
    return ModelBundle(calibrated, best, hi, med, meta), comparison


def holdout_metrics(bundle: ModelBundle, test_df: pd.DataFrame) -> dict:
    y = test_df[TARGET].to_numpy()
    p = bundle.predict_proba(test_df)
    order = np.argsort(-p)
    top10 = order[: max(1, len(p) // 10)]
    bins = np.clip((p * 10).astype(int), 0, 9)  # calibration: predicted vs observed per decile
    calib = [
        {"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int((bins == b).sum()),
         "predicted": round(float(p[bins == b].mean()), 3),
         "observed": round(float(y[bins == b].mean()), 3)}
        for b in range(10) if (bins == b).sum() >= 20
    ]  # fmt: skip
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "base_rate": round(float(y.mean()), 4),
        "precision_top_10pct": round(float(y[top10].mean()), 4),
        "lift_top_10pct": round(float(y[top10].mean() / y.mean()), 2),
        "recall_top_10pct": round(float(y[top10].sum() / y.sum()), 4),
        "calibration": calib,
    }
