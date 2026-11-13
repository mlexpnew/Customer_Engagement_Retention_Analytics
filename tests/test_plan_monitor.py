import numpy as np
import pandas as pd

from churnshield.monitor import drift_report, psi, segment_report
from churnshield.plan import Assumptions, compare_strategies, expected_net, realised_value, select


def frame(n=6):
    return pd.DataFrame({"Balance": [100_000.0] * n, "NumOfProducts": [2] * n})


def test_expected_value_formula():
    a = Assumptions(contact_cost=40, success_rate=0.3, margin_on_balance=0.01,
                    value_per_product=50, horizon_years=2)  # fmt: skip
    # value = (100000 * 0.01 + 2 * 50) * 2 = 2200; EV = 0.5 * 0.3 * 2200 - 40 = 290
    assert expected_net(np.array([0.5]), frame(1), a)[0] == 290


def test_select_respects_budget_and_skips_negative_value():
    a = Assumptions()
    p = np.array([0.9, 0.8, 0.001, 0.7, 0.6, 0.5])
    chosen = select(p, frame(), budget=a.contact_cost * 3, a=a)
    assert list(chosen) == [0, 1, 3]
    assert 2 not in select(p, frame(), budget=1e6, a=a)  # never pay to call a sure stayer


def test_realised_value_only_counts_real_churners():
    a = Assumptions()
    churned = np.array([1, 0, 0, 0, 0, 0])
    v = realised_value(np.array([0, 1]), frame(), churned, a)
    assert v == round(a.success_rate * a.value(frame(1))[0] - 2 * a.contact_cost, 6)


def test_strategy_comparison_shape():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"Balance": rng.uniform(0, 2e5, 300), "NumOfProducts": rng.integers(1, 4, 300)}
    )
    p = rng.uniform(0, 1, 300)
    churned = (rng.uniform(0, 1, 300) < p).astype(int)
    r = compare_strategies(p, df, churned, budget=2000, a=Assumptions())
    assert set(r["strategies"]) == {"churnshield_expected_value", "highest_churn_probability",
                                    "highest_balance", "random"}  # fmt: skip
    assert r["strategies"]["random"]["contacted"] == 50


def test_psi_detects_shift_and_ignores_noise(data):
    a, b = data["Age"].iloc[:5000], data["Age"].iloc[5000:]
    assert psi(a, b) < 0.1
    assert psi(a, b * 0.7) > 0.25
    assert psi(data["Geography"].iloc[:5000], data["Geography"].iloc[5000:]) < 0.1


def test_drift_report_flags_retraining(data):
    shifted = data.iloc[5000:].copy()
    shifted["Balance"] *= 0.4
    r = drift_report(data.iloc[:5000], shifted)
    assert r["features"]["Balance"]["status"] == "act" and r["retrain_recommended"]


def test_segment_report(data):
    p = np.full(len(data), 0.2)
    flagged = np.zeros(len(data), dtype=bool)
    r = segment_report(data, p, flagged, "Gender")
    assert set(r) == {"Female", "Male"} and r["Female"]["churn_rate"] > r["Male"]["churn_rate"]
