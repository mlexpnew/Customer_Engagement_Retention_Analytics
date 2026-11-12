import numpy as np
import pandas as pd
import pytest

from retainlens.data import FEATURES, TARGET, split, validate
from retainlens.model import choose, holdout_metrics


def test_dataset_is_clean_and_has_expected_shape(data):
    assert len(data) == 10_000 and data["CustomerId"].is_unique
    assert set(FEATURES) <= set(data.columns) and "Gender" not in FEATURES


def test_validation_rejects_with_reasons():
    bad = pd.DataFrame([
        {"CreditScore": 1200, "Geography": "France", "Age": 40, "Tenure": 2, "Balance": 0,
         "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 5e4},
        {"CreditScore": 650, "Geography": "", "Age": 12, "Tenure": 2, "Balance": 0,
         "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 5e4},
    ])  # fmt: skip
    v = validate(bad)
    assert v.valid.empty and len(v.rejected) == 2
    assert "CreditScore" in v.rejected.iloc[0]["reason"]
    assert "Age" in v.rejected.iloc[1]["reason"] and "Geography" in v.rejected.iloc[1]["reason"]


def test_validation_requires_columns():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"Age": [30]}))


def test_split_is_stratified_and_disjoint(data):
    tr, te = split(data)
    assert len(te) == 2000 and not set(tr.index) & set(te.index)
    assert abs(tr[TARGET].mean() - te[TARGET].mean()) < 0.005


def test_near_tie_prefers_simpler_model():
    cmp = {"logistic_regression": {"cv_roc_auc": 0.78}, "gradient_boosting": {"cv_roc_auc": 0.856},
           "random_forest": {"cv_roc_auc": 0.857}}  # fmt: skip
    assert choose(cmp) == "gradient_boosting"
    cmp["random_forest"]["cv_roc_auc"] = 0.87
    assert choose(cmp) == "random_forest"


def test_small_model_beats_chance_and_is_calibrated(small_bundle, data):
    _, te = split(data.sample(3000, random_state=9))
    m = holdout_metrics(small_bundle, te)
    assert m["roc_auc"] > 0.8 and m["lift_top_10pct"] > 2
    p = small_bundle.predict_proba(te)
    assert p.min() >= 0.01 and p.max() <= 0.99
    assert abs(p.mean() - te[TARGET].mean()) < 0.05  # calibrated on average
    assert set(np.unique(small_bundle.band(p))) <= {"low", "medium", "high"}
