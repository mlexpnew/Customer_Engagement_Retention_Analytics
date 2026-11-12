"""Retention planner: whom to contact with a fixed budget.

Expected net value of contacting a customer:

    churn probability x offer success rate x customer value  -  contact cost

Customers are ranked by that value and picked until the budget runs out. The success
rate and value formula are business assumptions, exposed as parameters, not learned:
without past campaign data (treatment and control), true uplift cannot be estimated,
and the README says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    contact_cost: float = 40.0  # per customer: RM time + offer cost, EUR
    success_rate: float = 0.30  # share of would-be churners the offer keeps
    margin_on_balance: float = 0.015  # annual net interest margin on balances
    value_per_product: float = 60.0  # annual fee income per product
    horizon_years: float = 3.0  # value horizon if retained

    def value(self, df: pd.DataFrame) -> np.ndarray:
        annual = df["Balance"].to_numpy() * self.margin_on_balance + (
            df["NumOfProducts"].to_numpy() * self.value_per_product
        )
        return annual * self.horizon_years


def expected_net(p: np.ndarray, df: pd.DataFrame, a: Assumptions) -> np.ndarray:
    return p * a.success_rate * a.value(df) - a.contact_cost


def select(p: np.ndarray, df: pd.DataFrame, budget: float, a: Assumptions) -> np.ndarray:
    """Indices to contact: positive expected value, best first, within budget."""
    ev = expected_net(p, df, a)
    order = np.argsort(-ev)
    n = int(budget // a.contact_cost)
    return np.array([i for i in order[:n] if ev[i] > 0], dtype=int)


def realised_value(
    chosen: np.ndarray, df: pd.DataFrame, churned: np.ndarray, a: Assumptions
) -> float:
    """Value on held-out customers using what actually happened: only real churners can be saved."""
    if len(chosen) == 0:
        return 0.0
    saved = churned[chosen] * a.success_rate * a.value(df.iloc[chosen])
    return float(saved.sum() - a.contact_cost * len(chosen))


def compare_strategies(p: np.ndarray, df: pd.DataFrame, churned: np.ndarray, budget: float,
                       a: Assumptions, seed: int = 0) -> dict:  # fmt: skip
    """Model-driven plan versus the rules banks commonly use instead."""
    n = int(budget // a.contact_cost)
    rng = np.random.default_rng(seed)
    strategies = {
        "retainlens_expected_value": select(p, df, budget, a),
        "highest_churn_probability": np.argsort(-p)[:n],
        "highest_balance": np.argsort(-df["Balance"].to_numpy())[:n],
        "random": rng.choice(len(df), size=min(n, len(df)), replace=False),
    }
    out = {}
    for name, idx in strategies.items():
        out[name] = {
            "contacted": int(len(idx)),
            "real_churners_reached": int(churned[idx].sum()) if len(idx) else 0,
            "net_value": round(realised_value(idx, df, churned, a), 0),
        }
    return {"budget": budget, "assumptions": asdict(a), "strategies": out}
