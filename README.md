# MM Engine

[![ci](https://github.com/parveen06naresh/MM/actions/workflows/ci.yml/badge.svg)](https://github.com/parveen06naresh/MM/actions/workflows/ci.yml)

A C++20 limit order book and matching engine with a Python research layer for market making, plus a derivatives module: Black-Scholes and Greeks, Monte Carlo with variance reduction, Heston and Merton jump-diffusion, and a delta-hedging simulator.

The engine is built for a zero-allocation hot path. Every result below comes from the scripts in this repo, and the numbers live in `results/`.

## Results

**Engine** (`results/bench_summary.json`; 5 runs x 5M messages, median; shared 4-vCPU Xeon 2.1 GHz cloud VM, GCC 13, -O3)

| Metric | Value |
| --- | --- |
| Throughput | 10.1M messages/s (range 9.3M-11.2M across runs) |
| Latency per message, median | 262 ns |
| Latency per message, p99 / p99.9 | 722 ns / 2.97 us |
| Timer overhead included in the latencies | ~21 ns |

The message mix is 50% limit orders (8% of them marketable), 35% cancels, 10% market orders and 5% size reductions. The book starts with 200,000 resting orders.

**Correctness** (`tests/`): 228 GoogleTest cases (207 for the book, 21 for options) pass with GCC and Clang, and again under AddressSanitizer and UBSan, on every push (see CI). 200 of them are randomized differential tests: each replays 4,000 random messages through both the engine and a simple `std::map` reference book. Every trade, best price, order count and level depth must match exactly.

**Avellaneda-Stoikov market making through the engine** (`results/as_mm.json`; 1,000 simulations, gamma = 0.1, the paper's parameters)

| Strategy | Mean P&L | P&L std | Mean / std | Final inventory std | 5th-percentile P&L |
| --- | --- | --- | --- | --- | --- |
| Inventory-aware (AS) | 64.41 | 6.72 | 9.58 | 2.96 | 53.52 |
| Symmetric around mid | 67.84 | 14.28 | 4.75 | 8.61 | 46.60 |

This reproduces the paper's main finding. Skewing quotes by inventory gives up about 5% of mean profit but halves P&L volatility, cuts inventory risk by about two thirds, and raises the worst-case (5th-percentile) outcome.

## Derivatives: pricing, Monte Carlo and hedging

`include/mme/options.hpp`, `src/options.cpp`. Every number below is in `results/options.json`, produced by `./build/options_report` (single thread, same VM as above). The base contract is S = K = 100, T = 1 year, r = 5%, q = 0, sigma = 20%.

**Black-Scholes.** Call 10.450584, put 5.573526 (the textbook values). Price plus all five Greeks takes about 75 ns. Tests check put-call parity on a grid, every Greek against central finite differences, the Black-Scholes PDE residual (theta + (r-q) S delta + 1/2 sigma^2 S^2 gamma - rV = 0, which is Ito's lemma plus no-arbitrage) to 1e-10, and implied volatility round trips (safeguarded Newton with a bisection fallback) across strikes from 50 to 200 and vols from 3% to 150%. `python/options_check.py` re-derives the prices with an independent NumPy implementation on 2,000 random contracts and runs in CI.

**Monte Carlo under GBM** (1M paths each, exact scheme)

| Method | Price | Std error | 95% CI | Variance reduction per path |
| --- | --- | --- | --- | --- |
| Plain | 10.4459 | 0.0147 | [10.4171, 10.4747] | 1.0x |
| Antithetic | 10.4465 | 0.0104 | [10.4262, 10.4669] | 2.0x |
| Control variate (discounted S_T) | 10.4474 | 0.0056 | [10.4364, 10.4584] | 6.9x |
| Antithetic + control variate | 10.4485 | 0.0027 | [10.4431, 10.4539] | 28.7x |

All four intervals contain the Black-Scholes price. Over 1,000 independent runs of 10,000 paths, the 95% interval covered the true price 95.0% of the time, so the error bars are honest. Error falls like 1/sqrt(paths): mean absolute error 0.427, 0.108, 0.036, 0.012 at 1k, 10k, 100k and 1M paths. Euler discretisation bias (400k antithetic + control-variate paths): -0.239 at 1 step, -0.038 at 4, -0.005 at 12, under one standard error by 252. The control variate uses the exact mean of the discretised scheme, so it adds no bias of its own.

**Delta hedging and Ito's lemma.** A dealer sells the call at the Black-Scholes price and delta-hedges N times over its life; the stock follows GBM with a 5% drift (20,000 paths per row).

| Rebalances N | Mean P&L | P&L std | Std as % of premium | Derman-Kamal estimate |
| --- | --- | --- | --- | --- |
| 4 | -0.012 | 3.202 | 30.6% | 3.326 |
| 16 | 0.003 | 1.659 | 15.9% | 1.663 |
| 52 (weekly) | -0.002 | 0.940 | 9.0% | 0.922 |
| 252 (daily) | -0.002 | 0.434 | 4.2% | 0.419 |
| 1,000 | 0.000 | 0.219 | 2.1% | 0.210 |

The mean is zero whatever the drift, which is the point of risk-neutral pricing. The fitted log-log slope of std against N is -0.486 (theory -0.5), and the measured std is within 5% of the Derman-Kamal approximation sqrt(pi/4) sigma vega / sqrt(N) from N = 16 on.

When the hedge uses the wrong volatility, Ito's lemma says each path earns the sum of 1/2 Gamma S^2 (sigma_hedge^2 dt - R^2) over the steps. At N = 1,000 (10,000 paths), hedging at 25% while realised vol is 20% earns a mean 1.988 against a predicted 1.987; hedging at 15% loses 1.961 against a predicted 1.962. Path by path, actual and predicted P&L have correlation 0.9995, with RMS gap 0.02-0.03 on a P&L std of 0.7-0.9.

**Stochastic volatility and jumps**

- *Heston*, semi-analytic via the characteristic function (the Albrecher et al. form that avoids the complex-log branch cut) and Gauss-Legendre quadrature: 5.785155434 on the Fang-Oosterlee benchmark (published reference 5.785155450, difference 1.6e-8). Full-truncation Euler Monte Carlo agrees within one standard error (0.023) at 52 and 252 steps; at 12 steps it is 0.064 high, which is discretisation bias. With rho = -0.7 the implied-vol smile falls from 26.2% at K = 70 to 14.1% at K = 130, the equity skew. With vol of vol near zero it reduces to Black-Scholes (tested).
- *Merton jump-diffusion* (lambda = 0.75 jumps a year, mean log jump -10%, jump vol 15%): series price 12.2244 against 10.4506 for Black-Scholes with the same diffusion vol; exact-sampling Monte Carlo gives 12.1958 with std error 0.0174 (1.6 standard errors away).

**Limits.** These are model results on simulated paths, not market data: no transaction costs, no discrete dividends, constant rates, and European exercise only. Heston and Merton parameters are illustrative, not calibrated to a real option chain. Calibrating Heston to a listed SPX or SPY chain is the natural next step.

## Design

- **Prices are integer ticks in a fixed band**, so a price level is one array lookup: no tree and no hashing on the price path.
- **Resting orders sit in a preallocated node pool** as an intrusive doubly linked FIFO per level (price-time priority). Add, fill and cancel never call the allocator.
- **Order-id lookup is an open-addressing hash table** with linear probing and backward-shift deletion. With no tombstones, lookups stay fast after millions of cancels.
- **The trade tape is reserved up front.** Python reads it as a zero-copy NumPy view through pybind11.
- **Operations:** limit, IOC, market, cancel, reduce (keeps queue position) and replace (loses it, like real exchanges).

## Layout

```
include/mme/order_book.hpp   engine API
src/order_book.cpp           matching, id map, level tracking
tests/                       unit tests + std::map reference book for differential tests
bench/bench_book.cpp         latency and throughput benchmark
python/bindings.cpp          pybind11 module (_mme)
python/as_mm.py              Avellaneda-Stoikov vs symmetric market maker
include/mme/options.hpp      derivatives API
src/options.cpp              Black-Scholes, Greeks, implied vol, Monte Carlo, Heston, Merton, hedging
bench/options_report.cpp     writes results/options.json
python/options_check.py      independent NumPy cross-check of the options module
scripts/bench_summary.py     multi-seed benchmark summary
```

## Build and run

```bash
cmake -S . -B build -G Ninja && cmake --build build
./build/mme_tests
python scripts/bench_summary.py > results/bench_summary.json

pip install pybind11 numpy
cmake -S . -B build-py -G Ninja -DMME_BUILD_PYTHON=ON -DMME_BUILD_TESTS=OFF && cmake --build build-py
python python/as_mm.py --sims 1000 > results/as_mm.json

./build/options_report > results/options.json
python python/options_check.py
```

## Limits (stated plainly)

- The benchmark stream is synthetic. Replaying real exchange data (LOBSTER, crypto L2 feeds) is the next step.
- Latencies are measured on a shared cloud VM with `std::chrono::steady_clock`. A pinned core on bare metal would be faster and less noisy.
- In the market-making simulation the strategy is the only liquidity, and taker arrivals follow the paper's exponential intensity, not real order flow.

## Roadmap

1. Replay real order-book data through the engine.
2. Calibrate a Hawkes process for order arrivals on that data, and quote against it.
3. Run every strategy through [Backtest Validator](https://github.com/parveen06naresh/backtest-validator) (multiple-testing control, overfitting and out-of-sample checks) before trusting a result.
4. Add a reinforcement-learning quoting policy, kept only if it beats Avellaneda-Stoikov after step 3.

## References

Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3), 217-224.

Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy* 81(3), 637-654.

Heston, S. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies* 6(2), 327-343.

Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83-92.

Fang, F. and Oosterlee, C. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing* 31(2), 826-848.

Merton, R. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics* 3(1-2), 125-144.

Derman, E. and Kamal, M. (1999). When you cannot hedge continuously: the corrections to Black-Scholes. *Risk* 12(1), 82-85.

Lord, R., Koekkoek, R. and van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. *Quantitative Finance* 10(2), 177-194.

Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

## License

MIT
