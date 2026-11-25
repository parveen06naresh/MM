#include <gtest/gtest.h>

#include <cmath>

#include "mme/options.hpp"

using namespace mme::opt;

namespace {
const Market kAtm{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};

double fd(const Market& m, Type t, double Market::*field, double h) {
  Market up = m, dn = m;
  up.*field += h;
  dn.*field -= h;
  return (bs_price(up, t) - bs_price(dn, t)) / (2.0 * h);
}
}  // namespace

// ------------------------------------------------------------------ Black-Scholes
TEST(BlackScholes, MatchesTextbookValues) {
  // Hull, Options Futures and Other Derivatives: S=K=100, T=1, r=5%, sigma=20%.
  EXPECT_NEAR(bs_price(kAtm, Type::Call), 10.450583572185565, 1e-12);
  EXPECT_NEAR(bs_price(kAtm, Type::Put), 5.573526022256971, 1e-12);
  // Hull example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81.
  const Market h{42.0, 40.0, 0.5, 0.10, 0.0, 0.2};
  EXPECT_NEAR(bs_price(h, Type::Call), 4.7594, 1e-4);
  EXPECT_NEAR(bs_price(h, Type::Put), 0.8086, 1e-4);
}

TEST(BlackScholes, PutCallParityAcrossGrid) {
  for (double S : {60.0, 90.0, 100.0, 130.0})
    for (double T : {0.05, 0.5, 2.0})
      for (double q : {0.0, 0.03})
        for (double sig : {0.05, 0.3, 0.9}) {
          const Market m{S, 100.0, T, 0.04, q, sig};
          const double lhs = bs_price(m, Type::Call) - bs_price(m, Type::Put);
          const double rhs = S * std::exp(-q * T) - 100.0 * std::exp(-0.04 * T);
          EXPECT_NEAR(lhs, rhs, 1e-10);
        }
}

TEST(BlackScholes, GreeksMatchFiniteDifferences) {
  for (Type t : {Type::Call, Type::Put})
    for (double S : {80.0, 100.0, 125.0})
      for (double q : {0.0, 0.02}) {
        const Market m{S, 100.0, 0.75, 0.03, q, 0.25};
        const Greeks g = bs_greeks(m, t);
        EXPECT_NEAR(g.delta, fd(m, t, &Market::S, 1e-4), 1e-7);
        EXPECT_NEAR(g.vega, fd(m, t, &Market::sigma, 1e-5), 1e-5);
        EXPECT_NEAR(g.rho, fd(m, t, &Market::r, 1e-5), 1e-5);
        EXPECT_NEAR(g.theta, -fd(m, t, &Market::T, 1e-5), 1e-5);  // theta = dV/dt = -dV/dT
        Market up = m, dn = m;
        up.S += 1e-2;
        dn.S -= 1e-2;
        const double gamma_fd = (bs_price(up, t) - 2.0 * bs_price(m, t) + bs_price(dn, t)) / 1e-4;
        EXPECT_NEAR(g.gamma, gamma_fd, 1e-6);
      }
}

TEST(BlackScholes, SatisfiesThePde) {
  // Theta + (r-q) S Delta + 0.5 sigma^2 S^2 Gamma - r V = 0: Ito's lemma plus no-arbitrage.
  for (Type t : {Type::Call, Type::Put})
    for (double S : {70.0, 100.0, 140.0}) {
      const Market m{S, 100.0, 1.3, 0.05, 0.01, 0.35};
      const Greeks g = bs_greeks(m, t);
      const double residual = g.theta + (m.r - m.q) * S * g.delta + 0.5 * m.sigma * m.sigma * S * S * g.gamma - m.r * g.price;
      EXPECT_NEAR(residual, 0.0, 1e-10);
    }
}

TEST(BlackScholes, ImpliedVolRoundTrips) {
  for (Type t : {Type::Call, Type::Put})
    for (double K : {50.0, 80.0, 100.0, 120.0, 200.0})
      for (double T : {0.02, 0.5, 3.0})
        for (double sig : {0.03, 0.15, 0.4, 1.5}) {
          Market m{100.0, K, T, 0.02, 0.01, sig};
          const double p = bs_price(m, t);
          const double lo = t == Type::Call ? std::max(100.0 * std::exp(-0.01 * T) - K * std::exp(-0.02 * T), 0.0)
                                            : std::max(K * std::exp(-0.02 * T) - 100.0 * std::exp(-0.01 * T), 0.0);
          if (p - lo < 1e-10) continue;  // no time value left to invert: vol is not identifiable
          m.sigma = 0.0;
          EXPECT_NEAR(implied_vol(p, m, t), sig, 1e-6) << "K=" << K << " T=" << T << " sig=" << sig;
        }
}

TEST(BlackScholes, ImpliedVolRejectsArbitragePrices) {
  EXPECT_EQ(implied_vol(150.0, kAtm, Type::Call), -1.0);  // above the spot
  EXPECT_EQ(implied_vol(1.0, kAtm, Type::Call), -1.0);    // below discounted intrinsic (4.88)
}

// ------------------------------------------------------------------ Monte Carlo
TEST(MonteCarlo, ExactSchemeAgreesWithBlackScholes) {
  const double bs = bs_price(kAtm, Type::Call);
  for (uint64_t seed = 1; seed <= 5; ++seed) {
    const McResult r = mc_price(kAtm, Type::Call, {200000, 1, Scheme::Exact, false, false, seed});
    EXPECT_LT(std::abs(r.price - bs), 4.0 * r.std_error) << "seed " << seed;
  }
}

TEST(MonteCarlo, ConfidenceIntervalCoverageIsNear95Percent) {
  const double bs = bs_price(kAtm, Type::Put);
  int covered = 0;
  const int trials = 400;
  for (int s = 0; s < trials; ++s) {
    const McResult r = mc_price(kAtm, Type::Put, {4000, 1, Scheme::Exact, false, false, 1000u + s});
    covered += (r.ci_low <= bs && bs <= r.ci_high);
  }
  const double cov = static_cast<double>(covered) / trials;
  EXPECT_GT(cov, 0.92);  // binomial sd at 400 trials is ~1.1 points
  EXPECT_LT(cov, 0.98);
}

TEST(MonteCarlo, VarianceReductionShrinksStandardError) {
  const McConfig base{100000, 1, Scheme::Exact, false, false, 7};
  McConfig anti = base, cv = base, both = base;
  anti.antithetic = true;
  cv.control_variate = true;
  both.antithetic = both.control_variate = true;
  const double bs = bs_price(kAtm, Type::Call);
  const McResult a = mc_price(kAtm, Type::Call, base), b = mc_price(kAtm, Type::Call, anti),
                 c = mc_price(kAtm, Type::Call, cv), d = mc_price(kAtm, Type::Call, both);
  EXPECT_LT(b.std_error, 0.75 * a.std_error);
  EXPECT_LT(c.std_error, 0.5 * a.std_error);
  EXPECT_LT(d.std_error, c.std_error);
  for (const McResult& r : {a, b, c, d}) EXPECT_LT(std::abs(r.price - bs), 4.0 * r.std_error);
}

TEST(MonteCarlo, EulerSchemeConvergesWithSteps) {
  const double bs = bs_price(kAtm, Type::Call);
  const McResult r = mc_price(kAtm, Type::Call, {100000, 100, Scheme::Euler, true, true, 3});
  EXPECT_LT(std::abs(r.price - bs), 4.0 * r.std_error + 0.01);
}

TEST(Rng, NormalsHaveUnitMomentsAndAreReproducible) {
  Rng a(42), b(42);
  double m1 = 0.0, m2 = 0.0;
  const int n = 1000000;
  for (int i = 0; i < n; ++i) {
    const double z = a.normal();
    EXPECT_EQ(z, b.normal());
    m1 += z;
    m2 += z * z;
  }
  EXPECT_NEAR(m1 / n, 0.0, 0.005);
  EXPECT_NEAR(m2 / n, 1.0, 0.005);
}

// ------------------------------------------------------------------ Heston
TEST(Heston, CollapsesToBlackScholesWhenVolOfVolIsZero) {
  const Heston h{0.04, 2.0, 0.04, 1e-4, 0.0};  // v0 = theta = 0.2^2, tiny xi
  for (double K : {80.0, 100.0, 120.0}) {
    Market m = kAtm;
    m.K = K;
    EXPECT_NEAR(heston_price(100.0, K, 1.0, 0.05, 0.0, h, Type::Call), bs_price(m, Type::Call), 1e-4);
  }
}

TEST(Heston, MatchesPublishedReferenceValue) {
  // Fang and Oosterlee (2008), COS method paper, Table 4 reference: 5.785155450.
  const Heston h{0.0175, 1.5768, 0.0398, 0.5751, -0.5711};
  EXPECT_NEAR(heston_price(100.0, 100.0, 1.0, 0.0, 0.0, h, Type::Call), 5.785155450, 1e-6);
}

TEST(Heston, PutCallParity) {
  const Heston h{0.05, 1.5, 0.04, 0.6, -0.7};
  const double c = heston_price(100.0, 110.0, 0.8, 0.03, 0.01, h, Type::Call);
  const double p = heston_price(100.0, 110.0, 0.8, 0.03, 0.01, h, Type::Put);
  EXPECT_NEAR(c - p, 100.0 * std::exp(-0.01 * 0.8) - 110.0 * std::exp(-0.03 * 0.8), 1e-9);
  EXPECT_GT(c, 0.0);
}

TEST(Heston, NegativeCorrelationProducesDownsideSkew) {
  // rho < 0: implied vol is higher for low strikes than for high strikes.
  const Heston h{0.04, 2.0, 0.04, 0.5, -0.7};
  Market m{100.0, 80.0, 1.0, 0.0, 0.0, 0.0};
  const double iv_low = implied_vol(heston_price(100.0, 80.0, 1.0, 0.0, 0.0, h, Type::Put), m, Type::Put);
  m.K = 120.0;
  const double iv_high = implied_vol(heston_price(100.0, 120.0, 1.0, 0.0, 0.0, h, Type::Call), m, Type::Call);
  EXPECT_GT(iv_low, iv_high + 0.02);
}

TEST(Heston, MonteCarloAgreesWithSemiAnalytic) {
  const Heston h{0.04, 2.0, 0.04, 0.5, -0.7};
  const double exact = heston_price(100.0, 100.0, 1.0, 0.03, 0.0, h, Type::Call);
  const McResult r = heston_mc(100.0, 100.0, 1.0, 0.03, 0.0, h, Type::Call, 100000, 100, 11);
  EXPECT_LT(std::abs(r.price - exact), 4.0 * r.std_error + 0.02);  // + small discretisation bias
}

// ------------------------------------------------------------------ Merton
TEST(Merton, NoJumpsIsBlackScholes) {
  EXPECT_NEAR(merton_price(kAtm, {0.0, -0.1, 0.2}, Type::Call), bs_price(kAtm, Type::Call), 1e-12);
}

TEST(Merton, MonteCarloAgreesWithSeries) {
  const Merton j{0.75, -0.1, 0.15};
  for (Type t : {Type::Call, Type::Put}) {
    const double exact = merton_price(kAtm, j, t);
    const McResult r = merton_mc(kAtm, j, t, 400000, 5);
    EXPECT_LT(std::abs(r.price - exact), 4.0 * r.std_error);
  }
  // Parity holds for the jump model too (it is a martingale model for the discounted spot).
  EXPECT_NEAR(merton_price(kAtm, j, Type::Call) - merton_price(kAtm, j, Type::Put), 100.0 - 100.0 * std::exp(-0.05), 1e-9);
}

// ------------------------------------------------------------------ delta hedging
TEST(DeltaHedge, ErrorShrinksLikeOneOverSqrtN) {
  HedgeConfig c;
  c.paths = 20000;
  c.rebalances = 16;
  const HedgeResult coarse = delta_hedge(kAtm, Type::Call, c);
  c.rebalances = 256;
  const HedgeResult fine = delta_hedge(kAtm, Type::Call, c);
  const double ratio = coarse.std_pnl / fine.std_pnl;  // theory: sqrt(256/16) = 4
  EXPECT_GT(ratio, 3.5);
  EXPECT_LT(ratio, 4.5);
  EXPECT_NEAR(fine.mean_pnl, 0.0, 4.0 * fine.std_pnl / std::sqrt(20000.0) + 0.01);
}

TEST(DeltaHedge, MatchesDermanKamalEstimate) {
  HedgeConfig c;
  c.rebalances = 100;
  const HedgeResult r = delta_hedge(kAtm, Type::Call, c);
  EXPECT_NEAR(r.std_pnl / r.derman_kamal_std, 1.0, 0.15);
}

TEST(DeltaHedge, ItoPredictionExplainsPathwisePnl) {
  // Hedge at 25% vol when realised is 20%: the seller earns 0.5*Gamma*S^2*(0.25^2-0.2^2)
  // per unit time, and Ito's lemma predicts each path's P&L.
  HedgeConfig c;
  c.sigma_hedge = 0.25;
  c.rebalances = 1000;
  c.paths = 4000;
  const HedgeResult r = delta_hedge(kAtm, Type::Call, c);
  EXPECT_GT(r.mean_pnl, 1.0);
  EXPECT_GT(r.corr_pnl_ito, 0.99);
  EXPECT_NEAR(r.mean_pnl, r.mean_ito_pnl, 0.05 * std::abs(r.mean_pnl));
}
