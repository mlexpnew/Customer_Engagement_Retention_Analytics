import pandas as pd
import pytest

from churnshield.cli import main
from churnshield.service import plan, score


def test_score_returns_reasons_and_actions(small_bundle, customers):
    scored, rejected = score(small_bundle, customers)
    assert len(scored) == 200 and rejected.empty
    top = scored.sort_values("churn_probability").iloc[-1]
    assert top["reasons"] and {"reason", "action", "risk_reduction"} <= set(top["reasons"][0])


def test_score_ignores_gender(small_bundle, customers):
    flipped = customers.copy()
    flipped["Gender"] = flipped["Gender"].map({"Male": "Female", "Female": "Male"})
    a, _ = score(small_bundle, customers)
    b, _ = score(small_bundle, flipped)
    assert (a["churn_probability"].to_numpy() == b["churn_probability"].to_numpy()).all()


def test_plan_ranks_by_expected_value_within_budget(small_bundle, customers):
    out = plan(small_bundle, customers, budget=400)
    assert len(out) <= 10 and out["expected_net_value"].is_monotonic_decreasing
    assert list(out["rank"]) == list(range(1, len(out) + 1))


def test_cli_score_and_plan(tmp_path, customers, capsys):
    src = tmp_path / "c.csv"
    customers.to_csv(src, index=False)
    assert main(["score", str(src), "--out", str(tmp_path / "s.csv")]) == 0
    assert main(["plan", str(src), "--budget", "800", "--out", str(tmp_path / "p.csv")]) == 0
    scored = pd.read_csv(tmp_path / "s.csv")
    assert {"churn_probability", "risk_band", "reasons", "actions"} <= set(scored.columns)


def test_api(customers):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from churnshield.api import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    rows = customers.head(20).to_dict(orient="records")
    r = c.post("/v1/score", json={"customers": rows})
    assert r.status_code == 200 and len(r.json()["scores"]) == 20
    r = c.post("/v1/plan", json={"customers": rows, "budget": 200})
    assert r.status_code == 200 and r.json()["spend"] <= 200
    assert c.post("/v1/score", json={"customers": []}).status_code == 422


def test_streamlit_app_runs():
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception and at.metric[0].value == "2,000"
