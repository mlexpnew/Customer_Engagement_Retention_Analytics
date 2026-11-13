"""Command line for analysts.

    churnshield score customers.csv --out scored.csv
    churnshield plan customers.csv --budget 20000 --out call_list.csv
    churnshield drift this_month.csv
"""

from __future__ import annotations

import argparse
import json
import sys

from .data import load
from .plan import Assumptions
from .service import drift, load_model, plan, score


def _flatten(df):
    out = df.copy()
    if "reasons" in out:
        out["reasons"] = out["reasons"].map(lambda rs: " | ".join(r["reason"] for r in rs))
        out["actions"] = df["reasons"].map(lambda rs: " | ".join(r["action"] for r in rs))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="churnshield")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("score", "plan", "drift"):
        s = sub.add_parser(name)
        s.add_argument("file", help="customers CSV or Excel")
        s.add_argument("--model", default=None)
        if name != "drift":
            s.add_argument("--out", required=True)
        if name == "plan":
            s.add_argument("--budget", type=float, required=True)
            s.add_argument("--contact-cost", type=float, default=40.0)
            s.add_argument("--success-rate", type=float, default=0.30)
    a = p.parse_args(argv)
    bundle = load_model(a.model) if a.model else load_model()
    df = load(a.file)

    if a.cmd == "score":
        scored, rejected = score(bundle, df)
        _flatten(scored).to_csv(a.out, index=False)
        print(f"scored {len(scored)}, rejected {len(rejected)}; "
              f"high risk {int((scored['risk_band'] == 'high').sum())} -> {a.out}")  # fmt: skip
        if len(rejected):
            print(rejected[["reason"]].head().to_string(), file=sys.stderr)
    elif a.cmd == "plan":
        asm = Assumptions(contact_cost=a.contact_cost, success_rate=a.success_rate)
        out = plan(bundle, df, a.budget, asm)
        _flatten(out).to_csv(a.out, index=False)
        print(f"contact {len(out)} customers, spend {len(out) * asm.contact_cost:,.0f}, "
              f"expected net value {out['expected_net_value'].sum():,.0f} -> {a.out}")  # fmt: skip
    else:
        print(json.dumps(drift(bundle, df), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
