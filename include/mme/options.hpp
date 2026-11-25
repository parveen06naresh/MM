// European option pricing: Black-Scholes closed form and Greeks, implied volatility,
// Monte Carlo under GBM with variance reduction, Heston (semi-analytic + Monte Carlo),
// Merton jump-diffusion (series + Monte Carlo), and a discrete delta-hedging simulator.
//
// Conventions: continuous rates r (risk-free) and q (dividend yield), T in years,
// sigma is annualised volatility. Theta is per year, vega and rho per unit (not per 1%).
#pragma once

#include <cstdint>
#include <vector>

namespace mme::opt {

enum class Type { Call, Put };

struct Market {
  double S;      // spot
  double K;      // strike
  double T;      // years to expiry
  double r;      // risk-free rate
  double q;      // dividend yield
  double sigma;  // volatility
};

struct Greeks {
  double price, delta, gamma, vega, theta, rho;
};

double norm_pdf(double x);
double norm_cdf(double x);

double bs_price(const Market& m, Type t);
Greeks bs_greeks(const Market& m, Type t);

// Implied volatility by safeguarded Newton (falls back to bisection). Returns -1 if the
// price is outside the no-arbitrage bounds.
double implied_vol(double price, Market m, Type t, double tol = 1e-12, int max_iter = 100);

// ---------------------------------------------------------------- random numbers
// xoshiro256** with Box-Muller normals: same stream on every compiler and stdlib,
// so results/ files are reproducible bit for bit.
class Rng {
 public:
  explicit Rng(uint64_t seed);
  uint64_t next();
  double uniform();  // (0, 1)
  double normal();
  double poisson(double mean);  // Knuth's method, fine for small means

 private:
  uint64_t s_[4];
  double spare_ = 0.0;
  bool has_spare_ = false;
};

// ---------------------------------------------------------------- Monte Carlo (GBM)
enum class Scheme { Exact, Euler };

struct McConfig {
  uint64_t paths = 100000;
  int steps = 1;  // time steps per path (Exact needs only 1 for a European payoff)
  Scheme scheme = Scheme::Exact;
  bool antithetic = false;
  bool control_variate = false;  // control: discounted S_T, known mean S*exp(-qT)
  uint64_t seed = 1;
};

struct McResult {
  double price;
  double std_error;
  double ci_low, ci_high;  // 95% confidence interval
  uint64_t samples;        // independent samples (antithetic pairs count once)
  double beta;             // control-variate coefficient (0 if unused)
};

McResult mc_price(const Market& m, Type t, const McConfig& cfg);

// ---------------------------------------------------------------- Heston
struct Heston {
  double v0;     // initial variance
  double kappa;  // mean-reversion speed
  double theta;  // long-run variance
  double xi;     // vol of vol
  double rho;    // spot/vol correlation
};

// Semi-analytic price via the characteristic function (Albrecher et al. 2007 form,
// which avoids the branch-cut discontinuity) and Gauss-Legendre quadrature.
double heston_price(double S, double K, double T, double r, double q, const Heston& h, Type t);

// Full-truncation Euler on the variance, log-Euler on the spot (Lord et al. 2010).
McResult heston_mc(double S, double K, double T, double r, double q, const Heston& h, Type t,
                   uint64_t paths, int steps, uint64_t seed);

// ---------------------------------------------------------------- Merton jump-diffusion
struct Merton {
  double lambda;  // jump intensity per year
  double mu_j;    // mean of log jump size
  double sig_j;   // std of log jump size
};

double merton_price(const Market& m, const Merton& j, Type t, int terms = 60);
McResult merton_mc(const Market& m, const Merton& j, Type t, uint64_t paths, uint64_t seed);

// ---------------------------------------------------------------- delta hedging
// A dealer sells one option at the Black-Scholes price under sigma_hedge, then
// delta-hedges with the stock `rebalances` times (evenly spaced) using Black-Scholes
// deltas at sigma_hedge. The stock follows GBM with drift mu and volatility sigma_real.
// Each path's P&L at expiry is cash + stock position - payoff.
//
// `ito_pnl` is the pathwise prediction from Ito's lemma / the gamma-theta identity:
//   sum_i e^{r(T-t_i)} * 0.5 * Gamma_i * S_i^2 * (sigma_hedge^2 * dt - R_i^2),
// with R_i the realised simple return over step i. As dt -> 0 the actual P&L converges
// to it path by path.
struct HedgeConfig {
  double mu = 0.05;
  double sigma_real = 0.2;
  double sigma_hedge = 0.2;
  int rebalances = 52;
  uint64_t paths = 20000;
  uint64_t seed = 1;
};

struct HedgeResult {
  double mean_pnl, std_pnl, mean_ito_pnl;
  double corr_pnl_ito;           // correlation of actual and predicted pathwise P&L
  double rmse_pnl_minus_ito;     // RMS of (actual - predicted)
  double derman_kamal_std;       // sqrt(pi/4) * sigma * vega / sqrt(N), the textbook estimate
};

HedgeResult delta_hedge(const Market& m, Type t, const HedgeConfig& cfg);

}  // namespace mme::opt
