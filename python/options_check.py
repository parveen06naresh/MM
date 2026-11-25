"""Cross-check the C++ derivatives module against an independent NumPy implementation.

    python python/options_check.py

Black-Scholes is re-derived here with math.erf, Monte Carlo is re-run with NumPy's own
generator, and the delta-hedging error is checked for the 1/sqrt(N) law. Exits non-zero
on any mismatch, so CI fails if the two implementations ever disagree.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build-py"))
import _mme as mme  # noqa: E402


def ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_numpy(S, K, T, r, q, sig, call=True):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if call:
        return S * math.exp(-q * T) * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)
    return K * math.exp(-r * T) * ncdf(-d2) - S * math.exp(-q * T) * ncdf(-d1)


failures = []


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


rng = np.random.default_rng(0)
worst = 0.0
for _ in range(2000):
    S, K = rng.uniform(50, 150), rng.uniform(50, 150)
    T, r, q, sig = rng.uniform(0.02, 3), rng.uniform(0, 0.08), rng.uniform(0, 0.04), rng.uniform(0.05, 0.9)
    call = bool(rng.integers(2))
    p = mme.bs_price(S, K, T, r, q, sig, call)
    worst = max(worst, abs(p - bs_numpy(S, K, T, r, q, sig, call)))
    fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
    time_value = p - max(fwd if call else -fwd, 0.0)
    if time_value > 1e-8:  # below this the price carries no information about vol
        iv = mme.implied_vol(p, S, K, T, r, q, call)
        worst = max(worst, abs(iv - sig))
check("Black-Scholes and implied vol vs NumPy (2,000 random contracts)", worst < 1e-6, f"max abs diff (price or vol) {worst:.2e}")

S, K, T, r, q, sig = 100.0, 105.0, 0.5, 0.04, 0.01, 0.3
exact = bs_numpy(S, K, T, r, q, sig)
z = rng.standard_normal(400_000)
ST = S * np.exp((r - q - 0.5 * sig**2) * T + sig * math.sqrt(T) * z)
x = math.exp(-r * T) * np.maximum(ST - K, 0.0)
np_se = x.std(ddof=1) / math.sqrt(len(x))
check("NumPy Monte Carlo vs Black-Scholes", abs(x.mean() - exact) < 4 * np_se, f"{x.mean():.4f} vs {exact:.4f} (se {np_se:.4f})")
cpp = mme.mc_price(S, K, T, r, q, sig, True, paths=200_000, antithetic=True, control_variate=True, seed=2)
check("C++ Monte Carlo (antithetic + control variate) vs Black-Scholes", abs(cpp["price"] - exact) < 4 * cpp["std_error"],
      f"{cpp['price']:.4f} vs {exact:.4f} (se {cpp['std_error']:.4f}, plain NumPy se at 2x paths {np_se:.4f})")

h16 = mme.delta_hedge(100, 100, 1, 0.05, 0, rebalances=16, paths=10_000)
h256 = mme.delta_hedge(100, 100, 1, 0.05, 0, rebalances=256, paths=10_000)
ratio = h16["std_pnl"] / h256["std_pnl"]
check("hedging error ratio N=16 vs N=256 (theory 4.0)", 3.5 < ratio < 4.5, f"{ratio:.2f}")

hv = mme.heston_price(100, 100, 1, 0, 0, 0.0175, 1.5768, 0.0398, 0.5751, -0.5711)
check("Heston vs published reference 5.785155450", abs(hv - 5.785155450) < 1e-6, f"{hv:.9f}")

sys.exit(1 if failures else 0)
