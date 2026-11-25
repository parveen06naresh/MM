// Replays a pre-generated synthetic message stream through the book and reports
// per-message latency percentiles and end-to-end throughput.
//   bench_book [n_messages] [seed] > results/bench.json
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "mme/order_book.hpp"

using namespace mme;
using clk = std::chrono::steady_clock;

enum class Op : std::uint8_t { Add, Cancel, Market, Reduce };
struct Msg { Op op; Side side; Price px; Qty qty; OrderId id; };

static std::vector<Msg> make_stream(std::size_t n, std::uint64_t seed, std::size_t warm) {
  std::mt19937_64 rng(seed);
  std::vector<Msg> out;
  out.reserve(n + warm);
  std::vector<OrderId> live;
  live.reserve(n + warm);
  double mid = 50'000;
  OrderId next = 1;
  std::normal_distribution<double> step(0.0, 0.6);
  std::geometric_distribution<int> offset(0.08);  // most orders rest near the touch
  auto add = [&](bool marketable_ok) {
    const Side s = (rng() & 1) ? Side::Buy : Side::Sell;
    int off = offset(rng);
    if (marketable_ok && rng() % 100 < 8) off = -1 - static_cast<int>(rng() % 3);  // crosses
    const Price px = static_cast<Price>(mid) + (s == Side::Buy ? -off - 1 : off + 1);
    out.push_back({Op::Add, s, px, 1 + static_cast<Qty>(rng() % 100), next});
    live.push_back(next++);
  };
  for (std::size_t i = 0; i < warm; ++i) add(false);
  for (std::size_t i = 0; i < n; ++i) {
    mid += step(rng);
    const unsigned r = static_cast<unsigned>(rng() % 100);
    if (r < 50 || live.empty()) {
      add(true);
    } else if (r < 85) {
      const std::size_t k = rng() % live.size();
      out.push_back({Op::Cancel, Side::Buy, 0, 0, live[k]});
      live[k] = live.back();
      live.pop_back();
    } else if (r < 95) {
      out.push_back({Op::Market, (rng() & 1) ? Side::Buy : Side::Sell, 0, 1 + static_cast<Qty>(rng() % 200), next++});
    } else {
      out.push_back({Op::Reduce, Side::Buy, 0, static_cast<Qty>(rng() % 50), live[rng() % live.size()]});
    }
  }
  return out;
}

static inline void apply(OrderBook& b, const Msg& m) {
  switch (m.op) {
    case Op::Add: b.add_limit(m.id, m.side, m.px, m.qty); break;
    case Op::Cancel: b.cancel(m.id); break;
    case Op::Market: b.add_market(m.id, m.side, m.qty); break;
    case Op::Reduce: b.reduce(m.id, m.qty); break;
  }
}

static double pct(std::vector<std::int64_t>& v, double p) {
  if (v.empty()) return 0;
  const std::size_t k = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
  return static_cast<double>(v[k]);
}

int main(int argc, char** argv) {
  const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 5'000'000;
  const std::uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 42;
  const std::size_t warm = 200'000;
  const auto stream = make_stream(n, seed, warm);
  BookConfig cfg{0, 100'000, static_cast<std::uint32_t>(warm + n), 1u << 16};

  // Pass 1: throughput, no per-message timer.
  double tput;
  {
    OrderBook b(cfg);
    for (std::size_t i = 0; i < warm; ++i) apply(b, stream[i]);
    const auto t0 = clk::now();
    for (std::size_t i = warm; i < stream.size(); ++i) {
      apply(b, stream[i]);
      if (b.trades().size() > 60'000) b.clear_trades();
    }
    const double s = std::chrono::duration<double>(clk::now() - t0).count();
    tput = static_cast<double>(n) / s;
  }
  // Pass 2: per-message latency (includes ~20 ns of clock overhead).
  std::vector<std::int64_t> lat[4];
  for (auto& v : lat) v.reserve(n);
  std::vector<std::int64_t> all;
  all.reserve(n);
  std::size_t resting_end = 0;
  {
    OrderBook b(cfg);
    for (std::size_t i = 0; i < warm; ++i) apply(b, stream[i]);
    for (std::size_t i = warm; i < stream.size(); ++i) {
      const auto t0 = clk::now();
      apply(b, stream[i]);
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
      lat[static_cast<int>(stream[i].op)].push_back(ns);
      all.push_back(ns);
      if (b.trades().size() > 60'000) b.clear_trades();
    }
    resting_end = b.order_count();
  }
  // Cost of one timer pair with no work in between, so readers can subtract it.
  std::vector<std::int64_t> ovh;
  ovh.reserve(1'000'000);
  for (int i = 0; i < 1'000'000; ++i) {
    const auto t0 = clk::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
    ovh.push_back(ns);
  }
  const char* names[4] = {"add", "cancel", "market", "reduce"};
  std::printf("{\n  \"messages\": %zu,\n  \"seed\": %llu,\n  \"warm_resting_orders\": %zu,\n  \"resting_orders_at_end\": %zu,\n",
              n, static_cast<unsigned long long>(seed), warm, resting_end);
  std::printf("  \"throughput_msgs_per_s\": %.0f,\n", tput);
  std::printf("  \"mean_ns_per_msg\": %.1f,\n", 1e9 / tput);
  std::printf("  \"timer_overhead_p50_ns\": %.0f,\n", pct(ovh, .5));
  auto row = [&](const char* name, std::vector<std::int64_t>& v, bool last) {
    std::printf("  \"%s\": {\"count\": %zu, \"p50_ns\": %.0f, \"p90_ns\": %.0f, \"p99_ns\": %.0f, \"p999_ns\": %.0f}%s\n",
                name, v.size(), pct(v, .5), pct(v, .9), pct(v, .99), pct(v, .999), last ? "" : ",");
  };
  row("all", all, false);
  for (int k = 0; k < 4; ++k) row(names[k], lat[k], k == 3);
  std::printf("}\n");
}
