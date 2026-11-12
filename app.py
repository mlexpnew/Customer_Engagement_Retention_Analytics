"""RetainLens demo (Streamlit): upload customers, get a risk list and a call plan.

    pip install -e ".[app]" && streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from retainlens.data import load  # noqa: E402
from retainlens.plan import Assumptions  # noqa: E402
from retainlens.service import drift, load_model, plan, score  # noqa: E402

st.set_page_config(page_title="RetainLens", page_icon="🏦", layout="wide")
st.title("RetainLens")
st.caption(
    "Which customers are about to leave, why, and who to call first with this month's budget."
)

bundle = load_model()
meta = bundle.metadata

with st.sidebar:
    st.subheader("Retention budget")
    budget = st.number_input("Budget (EUR)", 1000, 1_000_000, 20_000, step=1000)
    cost = st.number_input("Cost per contact (EUR)", 5, 500, 40)
    success = st.slider("Offer success rate", 0.05, 0.8, 0.30, 0.05)
    st.divider()
    st.caption(f"Model: {meta['model']} · held-out ROC-AUC {meta['holdout']['roc_auc']:.3f} · "
               f"trained {meta['trained_at'][:10]}")  # fmt: skip

up = st.file_uploader("Customers (CSV or Excel)", type=["csv", "xlsx"])
if up is None:
    df = load(Path(__file__).parent / "data" / "European_Bank.xlsx").sample(2000, random_state=7)
    df = df.drop(columns=["Exited"])
    st.info("Showing 2,000 sample customers. Upload your own file with the same columns.")
else:
    df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)

scored, rejected = score(bundle, df)
call = plan(bundle, df, budget, Assumptions(contact_cost=cost, success_rate=success))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers scored", f"{len(scored):,}")
c2.metric("High risk", f"{int((scored['risk_band'] == 'high').sum()):,}")
c3.metric("To contact", f"{len(call):,}")
value = call["expected_net_value"].sum() if len(call) else 0
c4.metric("Expected net value", f"EUR {value:,.0f}")

tab1, tab2, tab3 = st.tabs(["Call plan", "All customers", "Data health"])


def flat(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["why"] = d["reasons"].map(lambda rs: "; ".join(r["reason"] for r in rs))
    d["talking points"] = d["reasons"].map(lambda rs: "; ".join(r["action"] for r in rs))
    return d.drop(columns=["reasons"])


with tab1:
    if len(call):
        view = flat(call)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download call list (CSV)", view.to_csv(index=False), "call_list.csv")
    else:
        st.write("No customer has positive expected value at these settings.")
with tab2:
    st.dataframe(flat(scored).sort_values("churn_probability", ascending=False),
                 use_container_width=True, hide_index=True)  # fmt: skip
with tab3:
    d = drift(bundle, df)
    verdict = "retrain recommended" if d["retrain_recommended"] else "model inputs look stable"
    st.write(f"Overall drift status: **{d['overall']}** ({verdict})")
    st.dataframe(pd.DataFrame(d["features"]).T, use_container_width=True)
    if len(rejected):
        st.warning(f"{len(rejected)} rows rejected by validation")
        st.dataframe(rejected, use_container_width=True)
