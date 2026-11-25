# MM Engine

[![ci](https://github.com/parveen06naresh/MM/actions/workflows/ci.yml/badge.svg)](https://github.com/parveen06naresh/MM/actions/workflows/ci.yml)

A C++20 limit order book and matching engine with a Python research layer for market making.

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

**Correctness** (`tests/`): 207 GoogleTest cases pass with GCC and Clang, and again under AddressSanitizer and UBSan, on every push (see CI). 200 of them are randomized differential tests: each replays 4,000 random messages through both the engine and a simple `std::map` reference book. Every trade, best price, order count and level depth must match exactly.

**Avellaneda-Stoikov market making through the engine** (`results/as_mm.json`; 1,000 simulations, gamma = 0.1, the paper's parameters)

| Strategy | Mean P&L | P&L std | Mean / std | Final inventory std | 5th-percentile P&L |
| --- | --- | --- | --- | --- | --- |
| Inventory-aware (AS) | 64.41 | 6.72 | 9.58 | 2.96 | 53.52 |
| Symmetric around mid | 67.84 | 14.28 | 4.75 | 8.61 | 46.60 |

This reproduces the paper's main finding. Skewing quotes by inventory gives up about 5% of mean profit but halves P&L volatility, cuts inventory risk by about two thirds, and raises the worst-case (5th-percentile) outcome.

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

## License

MIT
