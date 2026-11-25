"""Avellaneda-Stoikov (2008) market making, executed through the C++ book.

The market maker posts one bid and one ask into the engine every step.
Market orders arrive with Poisson intensity A * exp(-k * delta), where delta
is the quote's distance from the mid, and fill against the resting quotes
inside the engine. We compare the inventory-aware AS strategy with the
symmetric strategy that quotes the same spread around the mid.

    python python/as_mm.py --sims 1000 > results/as_mm.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build-py"))
import _mme as mme  # noqa: E402

TICK = 0.01


def run(strategy, rng, s0=100.0, T=1.0, sigma=2.0, dt=0.005, gamma=0.1, k=1.5, A=140.0):
    book = mme.OrderBook(0, 40_000, 1024, 1 << 12)
    n = int(round(T / dt))
    s, q, cash = s0, 0, 0.0
    bid_id, ask_id, next_id = 0, 0, 1
    for i in range(n):
        tau = T - i * dt
        spread = gamma * sigma**2 * tau + (2.0 / gamma) * math.log(1.0 + gamma / k)
        center = s - q * gamma * sigma**2 * tau if strategy == "inventory" else s
        bid = math.floor((center - spread / 2) / TICK)
        ask = math.ceil((center + spread / 2) / TICK)
        if bid_id:
            book.cancel(bid_id)
        if ask_id:
            book.cancel(ask_id)
        bid_id, ask_id = next_id, next_id + 1
        next_id += 2
        book.add_limit(bid_id, mme.Side.Buy, bid, 1)
        book.add_limit(ask_id, mme.Side.Sell, ask, 1)
        # Taker flow: fill probability depends on distance from the mid.
        if rng.random() < A * math.exp(-k * (ask * TICK - s)) * dt:
            book.add_market(next_id, mme.Side.Buy, 1)
            next_id += 1
        if rng.random() < A * math.exp(-k * (s - bid * TICK)) * dt:
            book.add_market(next_id, mme.Side.Sell, 1)
            next_id += 1
        for t in book.trades():
            px = int(t["price"]) * TICK
            if int(t["taker_side"]) == 0:  # taker bought from our ask
                q -= int(t["qty"]); cash += px * int(t["qty"])
            else:
                q += int(t["qty"]); cash -= px * int(t["qty"])
        book.clear_trades()
        # Mid moves after quoting, as in the paper's discretisation.
        s += sigma * math.sqrt(dt) * (1 if rng.random() < 0.5 else -1)
    return cash + q * s, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gamma", type=float, default=0.1)
    a = ap.parse_args()
    out = {"params": {"s0": 100, "T": 1, "sigma": 2, "dt": 0.005, "gamma": a.gamma, "k": 1.5, "A": 140,
                      "tick": TICK, "sims": a.sims, "seed": a.seed}}
    for strat in ("inventory", "symmetric"):
        rng = np.random.default_rng(a.seed)
        res = np.array([run(strat, rng, gamma=a.gamma) for _ in range(a.sims)])
        pnl, q = res[:, 0], res[:, 1]
        out[strat] = {
            "pnl_mean": round(float(pnl.mean()), 2),
            "pnl_std": round(float(pnl.std(ddof=1)), 2),
            "pnl_mean_over_std": round(float(pnl.mean() / pnl.std(ddof=1)), 2),
            "final_inventory_mean": round(float(q.mean()), 2),
            "final_inventory_std": round(float(q.std(ddof=1)), 2),
            "pnl_5th_percentile": round(float(np.percentile(pnl, 5)), 2),
        }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
