// Prints every number quoted in the README's derivatives section as JSON.
//   ./build/options_report > results/options.json
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mme/options.hpp"

using namespace mme::opt;
using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

static void mc_row(const char* name, const Market& m, const McConfig& c, double bs, double plain_var_per_path,
                   bool last) {
  const auto t0 = Clock::now();
  const McResult r = mc_price(m, Type::Call, c);
  const double s = secs(t0);
  const double paths_used = static_cast<double>(c.paths) * (c.antithetic ? 2.0 : 1.0);
  const double var_per_path = r.std_error * r.std_error * paths_used;
  std::printf(
      "    {\"method\": \"%s\", \"paths\": %.0f, \"price\": %.6f, \"error_vs_bs\": %.6f, \"std_error\": %.6f, "
      "\"ci95\": [%.6f, %.6f], \"variance_reduction_per_path\": %.2f, \"seconds\": %.3f}%s\n",
      name, paths_used, r.price, r.price - bs, r.std_error, r.ci_low, r.ci_high,
      plain_var_per_path > 0 ? plain_var_per_path / var_per_path : 1.0, s, last ? "" : ",");
}

int main() {
  const Market m{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
  const double bs = bs_price(m, Type::Call);
  const Greeks g = bs_greeks(m, Type::Call);
  std::printf("{\n  \"market\": {\"S\": 100, \"K\": 100, \"T\": 1, \"r\": 0.05, \"q\": 0, \"sigma\": 0.2},\n");
  std::printf("  \"black_scholes\": {\"call\": %.10f, \"put\": %.10f, \"delta\": %.6f, \"gamma\": %.6f, "
              "\"vega\": %.6f, \"theta\": %.6f, \"rho\": %.6f},\n",
              bs, bs_price(m, Type::Put), g.delta, g.gamma, g.vega, g.theta, g.rho);

  {  // speed of closed-form price + all Greeks
    const int n = 2000000;
    double sink = 0.0;
    Market x = m;
    const auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) {
      x.S = 80.0 + 40.0 * (i % 1000) / 1000.0;
      sink += bs_greeks(x, Type::Call).delta;
    }
    const double ns = secs(t0) * 1e9 / n;
    std::fprintf(stderr, "sink %.3f\n", sink);
    std::printf("  \"bs_price_and_greeks_ns\": %.1f,\n", ns);
  }

  // Monte Carlo at 1M path-equivalents (antithetic uses 500k pairs = 1M paths).
  std::printf("  \"monte_carlo\": [\n");
  const McResult plain = mc_price(m, Type::Call, {1000000, 1, Scheme::Exact, false, false, 1});
  const double v0 = plain.std_error * plain.std_error * 1e6;
  mc_row("plain", m, {1000000, 1, Scheme::Exact, false, false, 1}, bs, v0, false);
  mc_row("antithetic", m, {500000, 1, Scheme::Exact, true, false, 1}, bs, v0, false);
  mc_row("control_variate", m, {1000000, 1, Scheme::Exact, false, true, 1}, bs, v0, false);
  mc_row("antithetic+control_variate", m, {500000, 1, Scheme::Exact, true, true, 1}, bs, v0, true);
  std::printf("  ],\n");

  {  // error vs paths: plain MC should shrink like 1/sqrt(n)
    std::printf("  \"convergence\": [");
    const std::vector<uint64_t> ns = {1000, 10000, 100000, 1000000};
    for (size_t i = 0; i < ns.size(); ++i) {
      double abs_err = 0.0, se = 0.0;
      const int reps = 20;
      for (int s = 0; s < reps; ++s) {
        const McResult r = mc_price(m, Type::Call, {ns[i], 1, Scheme::Exact, false, false, 500u + s});
        abs_err += std::abs(r.price - bs) / reps;
        se += r.std_error / reps;
      }
      std::printf("%s{\"paths\": %llu, \"mean_abs_error\": %.5f, \"mean_std_error\": %.5f}", i ? ", " : "",
                  static_cast<unsigned long long>(ns[i]), abs_err, se);
    }
    std::printf("],\n");
  }

  {  // 95% CI coverage over 1000 independent runs of 10k paths
    int covered = 0;
    const int trials = 1000;
    for (int s = 0; s < trials; ++s) {
      const McResult r = mc_price(m, Type::Call, {10000, 1, Scheme::Exact, false, false, 10000u + s});
      covered += r.ci_low <= bs && bs <= r.ci_high;
    }
    std::printf("  \"ci95_coverage\": {\"trials\": %d, \"paths_each\": 10000, \"covered\": %.3f},\n", trials,
                static_cast<double>(covered) / trials);
  }

  {  // Euler discretisation bias vs number of steps
    std::printf("  \"euler_bias\": [");
    const std::vector<int> steps = {1, 4, 12, 52, 252};
    for (size_t i = 0; i < steps.size(); ++i) {
      const McResult r = mc_price(m, Type::Call, {400000, steps[i], Scheme::Euler, true, true, 9});
      std::printf("%s{\"steps\": %d, \"error_vs_bs\": %.5f, \"std_error\": %.5f}", i ? ", " : "", steps[i],
                  r.price - bs, r.std_error);
    }
    std::printf("],\n");
  }

  {  // Heston
    const Heston ref{0.0175, 1.5768, 0.0398, 0.5751, -0.5711};
    const double hv = heston_price(100, 100, 1, 0, 0, ref, Type::Call);
    std::printf("  \"heston_reference\": {\"ours\": %.9f, \"published\": 5.785155450, \"abs_diff\": %.2e},\n", hv,
                std::abs(hv - 5.785155450));
    const Heston h{0.04, 2.0, 0.04, 0.5, -0.7};
    const double exact = heston_price(100, 100, 1, 0.03, 0, h, Type::Call);
    std::printf("  \"heston_mc\": {\"params\": {\"v0\": 0.04, \"kappa\": 2, \"theta\": 0.04, \"xi\": 0.5, \"rho\": -0.7, "
                "\"r\": 0.03}, \"semi_analytic\": %.6f, \"runs\": [",
                exact);
    const std::vector<int> steps = {12, 52, 252};
    for (size_t i = 0; i < steps.size(); ++i) {
      const McResult r = heston_mc(100, 100, 1, 0.03, 0, h, Type::Call, 200000, steps[i], 21);
      std::printf("%s{\"steps\": %d, \"paths\": 200000, \"error\": %.5f, \"std_error\": %.5f}", i ? ", " : "",
                  steps[i], r.price - exact, r.std_error);
    }
    std::printf("]},\n  \"heston_smile\": [");
    const std::vector<double> ks = {70, 80, 90, 100, 110, 120, 130};
    for (size_t i = 0; i < ks.size(); ++i) {
      const Type t = ks[i] < 100 ? Type::Put : Type::Call;
      Market mm{100, ks[i], 1, 0.03, 0, 0};
      const double iv = implied_vol(heston_price(100, ks[i], 1, 0.03, 0, h, t), mm, t);
      std::printf("%s{\"K\": %.0f, \"implied_vol\": %.4f}", i ? ", " : "", ks[i], iv);
    }
    std::printf("],\n");
  }

  {  // Merton
    const Merton j{0.75, -0.1, 0.15};
    const double exact = merton_price(m, j, Type::Call);
    const McResult r = merton_mc(m, j, Type::Call, 1000000, 4);
    std::printf("  \"merton\": {\"lambda\": 0.75, \"mu_j\": -0.1, \"sig_j\": 0.15, \"series\": %.6f, \"mc\": %.6f, "
                "\"mc_std_error\": %.6f, \"bs_same_diffusion_vol\": %.6f},\n",
                exact, r.price, r.std_error, bs);
  }

  {  // delta hedging
    std::printf("  \"delta_hedge\": {\"premium\": %.6f, \"vs_rebalances\": [", bs);
    const std::vector<int> ns = {4, 16, 52, 252, 1000};
    std::vector<double> lx, ly;
    for (size_t i = 0; i < ns.size(); ++i) {
      HedgeConfig c;
      c.rebalances = ns[i];
      c.paths = 20000;
      c.seed = 3;
      const HedgeResult h = delta_hedge(m, Type::Call, c);
      lx.push_back(std::log(ns[i]));
      ly.push_back(std::log(h.std_pnl));
      std::printf("%s{\"N\": %d, \"mean_pnl\": %.4f, \"std_pnl\": %.4f, \"std_pct_of_premium\": %.2f, "
                  "\"derman_kamal\": %.4f}",
                  i ? ", " : "", ns[i], h.mean_pnl, h.std_pnl, 100.0 * h.std_pnl / bs, h.derman_kamal_std);
    }
    double mx = 0, my = 0;
    for (size_t i = 0; i < lx.size(); ++i) mx += lx[i] / lx.size(), my += ly[i] / ly.size();
    double sxy = 0, sxx = 0;
    for (size_t i = 0; i < lx.size(); ++i) sxy += (lx[i] - mx) * (ly[i] - my), sxx += (lx[i] - mx) * (lx[i] - mx);
    std::printf("], \"loglog_slope\": %.4f, \"vol_mismatch\": [", sxy / sxx);
    const std::vector<double> sh = {0.15, 0.25};
    for (size_t i = 0; i < sh.size(); ++i) {
      HedgeConfig c;
      c.sigma_hedge = sh[i];
      c.rebalances = 1000;
      c.paths = 10000;
      c.seed = 4;
      const HedgeResult h = delta_hedge(m, Type::Call, c);
      std::printf("%s{\"sigma_hedge\": %.2f, \"sigma_real\": 0.20, \"N\": 1000, \"mean_pnl\": %.4f, "
                  "\"std_pnl\": %.4f, \"ito_predicted_mean\": %.4f, \"corr_actual_vs_ito\": %.5f, "
                  "\"rmse_actual_minus_ito\": %.4f}",
                  i ? ", " : "", sh[i], h.mean_pnl, h.std_pnl, h.mean_ito_pnl, h.corr_pnl_ito, h.rmse_pnl_minus_ito);
    }
    std::printf("]}\n}\n");
  }
  return 0;
}
