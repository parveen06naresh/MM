#include "mme/options.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>

namespace mme::opt {

namespace {

constexpr double kPi = std::numbers::pi;

double payoff(double ST, double K, Type t) { return t == Type::Call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0); }

uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

// Mean, unbiased variance, 95% CI from per-sample values.
McResult summarize(const std::vector<double>& x, double beta) {
  const double n = static_cast<double>(x.size());
  double mean = 0.0;
  for (double v : x) mean += v;
  mean /= n;
  double ss = 0.0;
  for (double v : x) ss += (v - mean) * (v - mean);
  const double se = std::sqrt(ss / (n - 1.0) / n);
  return {mean, se, mean - 1.959963984540054 * se, mean + 1.959963984540054 * se, x.size(), beta};
}

// Gauss-Legendre nodes/weights on [-1, 1] (Newton on P_n).
template <int N>
struct GaussLegendre {
  std::array<double, N> x{}, w{};
  GaussLegendre() {
    for (int i = 0; i < N; ++i) {
      double z = std::cos(kPi * (i + 0.75) / (N + 0.5)), pp = 0.0;
      for (int it = 0; it < 100; ++it) {
        double p1 = 1.0, p2 = 0.0;
        for (int j = 1; j <= N; ++j) {
          const double p3 = p2;
          p2 = p1;
          p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
        }
        pp = N * (z * p1 - p2) / (z * z - 1.0);
        const double dz = p1 / pp;
        z -= dz;
        if (std::abs(dz) < 1e-15) break;
      }
      x[i] = z;
      w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
  }
};

}  // namespace

// ---------------------------------------------------------------- Black-Scholes
double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); }
double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }

Greeks bs_greeks(const Market& m, Type t) {
  const double dq = std::exp(-m.q * m.T), dr = std::exp(-m.r * m.T);
  if (m.T <= 0.0 || m.sigma <= 0.0) {
    // Expired or zero-vol: value is the discounted intrinsic value of the forward.
    const double fwd = m.S * dq - m.K * dr;
    const bool itm = t == Type::Call ? fwd > 0.0 : fwd < 0.0;
    const double price = itm ? std::abs(fwd) : 0.0;
    const double delta = itm ? (t == Type::Call ? dq : -dq) : 0.0;
    return {price, delta, 0.0, 0.0, 0.0, 0.0};
  }
  const double sq = std::sqrt(m.T);
  const double d1 = (std::log(m.S / m.K) + (m.r - m.q + 0.5 * m.sigma * m.sigma) * m.T) / (m.sigma * sq);
  const double d2 = d1 - m.sigma * sq;
  const double nd1 = norm_pdf(d1);
  Greeks g{};
  g.gamma = dq * nd1 / (m.S * m.sigma * sq);
  g.vega = m.S * dq * nd1 * sq;
  const double decay = -m.S * dq * nd1 * m.sigma / (2.0 * sq);
  if (t == Type::Call) {
    g.price = m.S * dq * norm_cdf(d1) - m.K * dr * norm_cdf(d2);
    g.delta = dq * norm_cdf(d1);
    g.theta = decay - m.r * m.K * dr * norm_cdf(d2) + m.q * m.S * dq * norm_cdf(d1);
    g.rho = m.K * m.T * dr * norm_cdf(d2);
  } else {
    g.price = m.K * dr * norm_cdf(-d2) - m.S * dq * norm_cdf(-d1);
    g.delta = -dq * norm_cdf(-d1);
    g.theta = decay + m.r * m.K * dr * norm_cdf(-d2) - m.q * m.S * dq * norm_cdf(-d1);
    g.rho = -m.K * m.T * dr * norm_cdf(-d2);
  }
  return g;
}

double bs_price(const Market& m, Type t) { return bs_greeks(m, t).price; }

double implied_vol(double price, Market m, Type t, double tol, int max_iter) {
  const double dq = std::exp(-m.q * m.T), dr = std::exp(-m.r * m.T);
  const double lo_bound = t == Type::Call ? std::max(m.S * dq - m.K * dr, 0.0) : std::max(m.K * dr - m.S * dq, 0.0);
  const double hi_bound = t == Type::Call ? m.S * dq : m.K * dr;
  if (!(price > lo_bound && price < hi_bound)) return -1.0;
  double lo = 1e-9, hi = 10.0, s = 0.2;
  for (int i = 0; i < max_iter; ++i) {
    m.sigma = s;
    const Greeks g = bs_greeks(m, t);
    const double diff = g.price - price;
    if (std::abs(diff) < tol) return s;
    (diff > 0.0 ? hi : lo) = s;  // price is increasing in sigma
    double next = g.vega > 1e-14 ? s - diff / g.vega : 0.5 * (lo + hi);
    if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);  // Newton left the bracket: bisect
    if (std::abs(next - s) < 1e-15) return next;
    s = next;
  }
  return s;
}

// ---------------------------------------------------------------- RNG
Rng::Rng(uint64_t seed) {
  for (auto& w : s_) w = splitmix64(seed);
}

uint64_t Rng::next() {
  const uint64_t result = rotl(s_[1] * 5, 7) * 9;
  const uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

double Rng::uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }

double Rng::normal() {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }
  const double r = std::sqrt(-2.0 * std::log(uniform()));
  const double a = 2.0 * kPi * uniform();
  spare_ = r * std::sin(a);
  has_spare_ = true;
  return r * std::cos(a);
}

double Rng::poisson(double mean) {
  const double L = std::exp(-mean);
  int k = 0;
  double p = 1.0;
  do {
    ++k;
    p *= uniform();
  } while (p > L);
  return k - 1;
}

// ---------------------------------------------------------------- Monte Carlo (GBM)
McResult mc_price(const Market& m, Type t, const McConfig& cfg) {
  Rng rng(cfg.seed);
  const int steps = std::max(cfg.steps, 1);
  const double dt = m.T / steps, sdt = std::sqrt(dt), disc = std::exp(-m.r * m.T);
  const double drift_exact = (m.r - m.q - 0.5 * m.sigma * m.sigma) * dt;
  const double drift_euler = (m.r - m.q) * dt;
  std::vector<double> z(steps);

  auto terminal = [&](double sign) {
    double S = m.S;
    if (cfg.scheme == Scheme::Exact) {
      double logret = 0.0;
      for (int k = 0; k < steps; ++k) logret += drift_exact + m.sigma * sdt * sign * z[k];
      S *= std::exp(logret);
    } else {
      for (int k = 0; k < steps; ++k) S += S * (drift_euler + m.sigma * sdt * sign * z[k]);
    }
    return S;
  };

  // Known mean of the control (discounted S_T) under the scheme actually simulated.
  const double control_mean = cfg.scheme == Scheme::Exact ? m.S * std::exp(-m.q * m.T)
                                                          : disc * m.S * std::pow(1.0 + drift_euler, steps);
  std::vector<double> x(cfg.paths), y(cfg.paths);
  for (uint64_t i = 0; i < cfg.paths; ++i) {
    for (auto& v : z) v = rng.normal();
    const double s1 = terminal(1.0);
    double xv = disc * payoff(s1, m.K, t), yv = disc * s1;
    if (cfg.antithetic) {
      const double s2 = terminal(-1.0);
      xv = 0.5 * (xv + disc * payoff(s2, m.K, t));
      yv = 0.5 * (yv + disc * s2);
    }
    x[i] = xv;
    y[i] = yv;
  }
  if (!cfg.control_variate) return summarize(x, 0.0);

  const double n = static_cast<double>(cfg.paths);
  double mx = 0.0, my = 0.0;
  for (uint64_t i = 0; i < cfg.paths; ++i) mx += x[i], my += y[i];
  mx /= n;
  my /= n;
  double cxy = 0.0, vyy = 0.0;
  for (uint64_t i = 0; i < cfg.paths; ++i) cxy += (x[i] - mx) * (y[i] - my), vyy += (y[i] - my) * (y[i] - my);
  const double beta = cxy / vyy;
  for (uint64_t i = 0; i < cfg.paths; ++i) x[i] -= beta * (y[i] - control_mean);
  return summarize(x, beta);
}

// ---------------------------------------------------------------- Heston
double heston_price(double S, double K, double T, double r, double q, const Heston& h, Type t) {
  using cd = std::complex<double>;
  const cd i(0.0, 1.0);
  const double lnS = std::log(S), lnK = std::log(K);
  // Characteristic function of ln S_T (Albrecher, Mayer, Schoutens, Tistaert 2007).
  auto phi = [&](cd u) {
    const cd a = h.kappa - h.rho * h.xi * i * u;
    const cd d = std::sqrt(a * a + h.xi * h.xi * (i * u + u * u));
    const cd g = (a - d) / (a + d);
    const cd e = std::exp(-d * T);
    const cd C = i * u * (lnS + (r - q) * T) +
                 h.kappa * h.theta / (h.xi * h.xi) * ((a - d) * T - 2.0 * std::log((1.0 - g * e) / (1.0 - g)));
    const cd D = (a - d) / (h.xi * h.xi) * (1.0 - e) / (1.0 - g * e);
    return std::exp(C + D * h.v0);
  };
  static const GaussLegendre<32> gl;
  const cd phi_mi = phi(cd(0.0, -1.0));  // = S e^{(r-q)T}
  const double U = 400.0;
  const int panels = 80;
  const double width = U / panels;
  double P1 = 0.0, P2 = 0.0;
  for (int p = 0; p < panels; ++p) {
    const double a = p * width;
    for (int k = 0; k < 32; ++k) {
      const double u = a + 0.5 * width * (gl.x[k] + 1.0);
      const double w = 0.5 * width * gl.w[k];
      const cd eiu = std::exp(-i * u * lnK) / (i * u);
      P1 += w * std::real(eiu * phi(cd(u, -1.0)) / phi_mi);
      P2 += w * std::real(eiu * phi(cd(u, 0.0)));
    }
  }
  P1 = 0.5 + P1 / kPi;
  P2 = 0.5 + P2 / kPi;
  const double call = S * std::exp(-q * T) * P1 - K * std::exp(-r * T) * P2;
  return t == Type::Call ? call : call - S * std::exp(-q * T) + K * std::exp(-r * T);
}

McResult heston_mc(double S, double K, double T, double r, double q, const Heston& h, Type t, uint64_t paths,
                   int steps, uint64_t seed) {
  Rng rng(seed);
  const double dt = T / steps, sdt = std::sqrt(dt), disc = std::exp(-r * T);
  const double rho_c = std::sqrt(1.0 - h.rho * h.rho);
  std::vector<double> x(paths);
  for (uint64_t p = 0; p < paths; ++p) {
    double lnS = std::log(S), v = h.v0;
    for (int k = 0; k < steps; ++k) {
      const double z1 = rng.normal(), z2 = h.rho * z1 + rho_c * rng.normal();
      const double vp = std::max(v, 0.0);
      lnS += (r - q - 0.5 * vp) * dt + std::sqrt(vp) * sdt * z1;
      v += h.kappa * (h.theta - vp) * dt + h.xi * std::sqrt(vp) * sdt * z2;
    }
    x[p] = disc * payoff(std::exp(lnS), K, t);
  }
  return summarize(x, 0.0);
}

// ---------------------------------------------------------------- Merton
double merton_price(const Market& m, const Merton& j, Type t, int terms) {
  const double k = std::exp(j.mu_j + 0.5 * j.sig_j * j.sig_j) - 1.0;
  const double lam = j.lambda * (1.0 + k);
  if (lam * m.T <= 0.0) return bs_price(m, t);
  double total = 0.0;
  for (int n = 0; n < terms; ++n) {
    const double logw = -lam * m.T + n * std::log(lam * m.T) - std::lgamma(n + 1.0);
    Market mn = m;
    mn.sigma = std::sqrt(m.sigma * m.sigma + n * j.sig_j * j.sig_j / m.T);
    mn.r = m.r - j.lambda * k + n * std::log1p(k) / m.T;
    // bs_price discounts with mn.r; Merton's series uses that same r_n, so no correction.
    total += std::exp(logw) * bs_price(mn, t);
  }
  return total;
}

McResult merton_mc(const Market& m, const Merton& j, Type t, uint64_t paths, uint64_t seed) {
  Rng rng(seed);
  const double k = std::exp(j.mu_j + 0.5 * j.sig_j * j.sig_j) - 1.0;
  const double drift = (m.r - m.q - j.lambda * k - 0.5 * m.sigma * m.sigma) * m.T;
  const double disc = std::exp(-m.r * m.T), sq = std::sqrt(m.T);
  std::vector<double> x(paths);
  for (uint64_t p = 0; p < paths; ++p) {
    const double n = rng.poisson(j.lambda * m.T);
    const double jumps = n * j.mu_j + j.sig_j * std::sqrt(n) * rng.normal();
    const double ST = m.S * std::exp(drift + m.sigma * sq * rng.normal() + jumps);
    x[p] = disc * payoff(ST, m.K, t);
  }
  return summarize(x, 0.0);
}

// ---------------------------------------------------------------- delta hedging
HedgeResult delta_hedge(const Market& m0, Type t, const HedgeConfig& cfg) {
  Rng rng(cfg.seed);
  const int N = cfg.rebalances;
  const double dt = m0.T / N, sdt = std::sqrt(dt);
  const double growth = std::exp(m0.r * dt), div_yield = std::exp(m0.q * dt) - 1.0;
  Market mh = m0;
  mh.sigma = cfg.sigma_hedge;
  const Greeks g0 = bs_greeks(mh, t);
  const double sh2 = cfg.sigma_hedge * cfg.sigma_hedge;
  const double drift = (cfg.mu - m0.q - 0.5 * cfg.sigma_real * cfg.sigma_real) * dt;

  std::vector<double> pnl(cfg.paths), ito(cfg.paths);
  for (uint64_t p = 0; p < cfg.paths; ++p) {
    double S = m0.S, delta = g0.delta, gamma = g0.gamma;
    double cash = g0.price - delta * S;  // sold the option, bought delta shares
    double pred = 0.0;
    for (int i = 0; i < N; ++i) {
      const double S_next = S * std::exp(drift + cfg.sigma_real * sdt * rng.normal());
      cash = cash * growth + delta * S * div_yield;  // interest, plus dividends on the shares held
      const double R = S_next / S - 1.0;
      pred += std::exp(m0.r * (m0.T - (i + 1) * dt)) * 0.5 * gamma * S * S * (sh2 * dt - R * R);
      S = S_next;
      if (i + 1 < N) {
        mh.S = S;
        mh.T = m0.T - (i + 1) * dt;
        const Greeks g = bs_greeks(mh, t);
        cash -= (g.delta - delta) * S;
        delta = g.delta;
        gamma = g.gamma;
      }
    }
    pnl[p] = cash + delta * S - payoff(S, m0.K, t);
    ito[p] = pred;
  }
  const McResult a = summarize(pnl, 0.0), b = summarize(ito, 0.0);
  const double n = static_cast<double>(cfg.paths);
  double cov = 0.0, va = 0.0, vb = 0.0, sq = 0.0;
  for (uint64_t p = 0; p < cfg.paths; ++p) {
    cov += (pnl[p] - a.price) * (ito[p] - b.price);
    va += (pnl[p] - a.price) * (pnl[p] - a.price);
    vb += (ito[p] - b.price) * (ito[p] - b.price);
    sq += (pnl[p] - ito[p]) * (pnl[p] - ito[p]);
  }
  Market mr = m0;
  mr.sigma = cfg.sigma_real;
  return {a.price,
          std::sqrt(va / (n - 1.0)),
          b.price,
          cov / std::sqrt(va * vb),
          std::sqrt(sq / n),
          std::sqrt(kPi / 4.0) * cfg.sigma_real * bs_greeks(mr, t).vega / std::sqrt(static_cast<double>(N))};
}

}  // namespace mme::opt
