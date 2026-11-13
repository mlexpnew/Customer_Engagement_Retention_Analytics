"""Train ChurnShield end to end and write the model card numbers.

    python -m churnshield.train --data data/European_Bank.xlsx --out models --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import FEATURES, PROTECTED, TARGET, add_features, load, split, validate
from .explain import global_importance, reference_profile
from .model import SEED, candidates, holdout_metrics, train
from .monitor import drift_report, segment_report
from .plan import Assumptions, compare_strategies

GATES = {"roc_auc": 0.84, "lift_top_10pct": 2.5}


def gender_ablation(train_df) -> dict:
    """What would using Gender as an input buy? Measured, then declined."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from .data import CATEGORICAL, ENGINEERED, NUMERIC

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    y = train_df[TARGET]
    without = cross_val_score(candidates()["gradient_boosting"], train_df[FEATURES], y,
                              cv=cv, scoring="roc_auc").mean()  # fmt: skip
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC + ENGINEERED),
                             ("cat", OneHotEncoder(handle_unknown="ignore"),
                              CATEGORICAL + PROTECTED)])  # fmt: skip
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=SEED)  # fmt: skip
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    with_g = cross_val_score(
        pipe, train_df[FEATURES + PROTECTED], y, cv=cv, scoring="roc_auc"
    ).mean()
    return {"cv_auc_without_gender": round(without, 4), "cv_auc_with_gender": round(with_g, 4),
            "auc_cost_of_excluding": round(with_g - without, 4)}  # fmt: skip


def shifted_batch(df, seed: int = 1):
    """A plausible bad month: younger, lower-balance customers arrive after a campaign."""
    rng = np.random.default_rng(seed)
    b = df.sample(1500, random_state=seed).copy()
    b["Age"] = (b["Age"] * 0.75).round().clip(18)
    b["Balance"] = b["Balance"] * rng.uniform(0.3, 0.7, len(b))
    return b


def report_markdown(r: dict) -> str:
    h, cmp = r["holdout"], r["cv_comparison"]
    lines = [
        f"### ChurnShield model report ({r['rows']['train']} train / {r['rows']['test']} "
        f"held-out customers, {r['rows']['rejected']} rejected by validation)",
        "",
        "| Candidate (5-fold CV on train) | ROC-AUC |",
        "|---|---|",
        *[f"| {k}{' (selected)' if k == r['model'] else ''} | {v['cv_roc_auc']:.3f} "
          f"± {v['cv_std']:.3f} |" for k, v in cmp.items()],
        "",
        "| Held-out metric | Value |",
        "|---|---|",
        f"| ROC-AUC | {h['roc_auc']:.3f} |",
        f"| PR-AUC (base rate {h['base_rate']:.1%}) | {h['pr_auc']:.3f} |",
        f"| Brier score (calibrated) | {h['brier']:.3f} |",
        f"| Top 10% riskiest: share who actually left | {h['precision_top_10pct']:.1%} "
        f"({h['lift_top_10pct']}x the base rate) |",
        f"| Top 10% riskiest: share of all leavers caught | {h['recall_top_10pct']:.1%} |",
        "",
        "| Calibration bin | Customers | Predicted | Observed |",
        "|---|---|---|---|",
        *[f"| {c['bin']} | {c['n']} | {c['predicted']:.2f} | {c['observed']:.2f} |"
          for c in h["calibration"]],
        "",
        f"**Retention budget test** (EUR {r['plan']['budget']:,.0f} on held-out customers, "
        f"realised with actual outcomes)",
        "",
        "| Strategy | Contacted | Real leavers reached | Net value (EUR) |",
        "|---|---|---|---|",
        *[f"| {k} | {v['contacted']} | {v['real_churners_reached']} | {v['net_value']:,.0f} |"
          for k, v in r["plan"]["strategies"].items()],
        "",
        "**Fairness**: gender is not a model input. Using it would add "
        f"{r['gender_ablation']['auc_cost_of_excluding']:+.4f} CV ROC-AUC.",
        "",
        "| Gender | Customers | Churn rate | ROC-AUC | Flagged | Leavers caught |",
        "|---|---|---|---|---|---|",
        *[f"| {k} | {v['customers']} | {v['churn_rate']:.1%} | {v['roc_auc']} | "
          f"{v['flagged_rate']:.1%} | {v['recall_of_churners']:.1%} |"
          for k, v in r["fairness"]["Gender"].items()],
        "",
        "| Geography | Customers | Churn rate | ROC-AUC | Flagged | Leavers caught |",
        "|---|---|---|---|---|---|",
        *[f"| {k} | {v['customers']} | {v['churn_rate']:.1%} | {v['roc_auc']} | "
          f"{v['flagged_rate']:.1%} | {v['recall_of_churners']:.1%} |"
          for k, v in r["fairness"]["Geography"].items()],
        "",
        "| Top churn drivers (mean risk reduction if fixed) | |",
        "|---|---|",
        *[f"| {k} | {v:.3f} |" for k, v in r["drivers"].items()],
        "",
        f"**Drift check**: held-out month `{r['drift']['holdout']}`, "
        f"simulated shifted month `{r['drift']['shifted']}` "
        f"(worst PSI {r['drift']['shifted_worst_psi']:.2f}, retrain recommended).",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train and evaluate ChurnShield")
    p.add_argument("--data", default="data/European_Bank.xlsx")
    p.add_argument("--out", default="models")
    p.add_argument("--budget", type=float, default=20000.0, help="retention budget for the test")
    p.add_argument("--gate", action="store_true", help="exit 1 below quality gates")
    a = p.parse_args(argv)

    raw = load(a.data)
    v = validate(raw)
    df = add_features(v.valid)
    train_df, test_df = split(df)
    bundle, comparison = train(train_df)
    hold = holdout_metrics(bundle, test_df)

    p_test = bundle.predict_proba(test_df)
    flagged = bundle.band(p_test) != "low"
    ref = reference_profile(train_df)
    plan = compare_strategies(p_test, test_df, test_df[TARGET].to_numpy(), a.budget, Assumptions())
    drift_hold = drift_report(train_df, test_df)
    shifted = drift_report(train_df, shifted_batch(test_df))
    r = {
        "model": bundle.name,
        "rows": {"train": len(train_df), "test": len(test_df), "rejected": len(v.rejected)},
        "cv_comparison": comparison,
        "holdout": hold,
        "plan": plan,
        "gender_ablation": gender_ablation(train_df),
        "fairness": {
            "Gender": segment_report(test_df, p_test, flagged, "Gender"),
            "Geography": segment_report(test_df, p_test, flagged, "Geography"),
        },
        "drivers": global_importance(bundle, test_df, ref),
        "drift": {
            "holdout": drift_hold["overall"],
            "shifted": shifted["overall"],
            "shifted_worst_psi": max(f["psi"] for f in shifted["features"].values()),
        },
    }
    bundle.metadata.update(holdout=hold, reference_profile=ref)
    bundle.reference = train_df.sample(2000, random_state=SEED)  # for drift checks in the API
    out = Path(a.out)
    bundle.save(out / "churnshield.joblib")
    (out / "metrics.json").write_text(json.dumps(r, indent=2, default=str))
    md = report_markdown(r)
    (out / "report.md").write_text(md)
    print(md)
    if a.gate:
        failed = [k for k, t in GATES.items() if hold[k] < t]
        if failed:
            print(f"below gate: {failed}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
