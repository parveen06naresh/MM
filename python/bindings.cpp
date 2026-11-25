// pybind11 bridge. Depth snapshots and the trade tape come back as NumPy arrays;
// the trade tape is a zero-copy view into the engine's own buffer.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mme/options.hpp"
#include "mme/order_book.hpp"

namespace py = pybind11;
using namespace mme;

PYBIND11_MODULE(_mme, m) {
  m.doc() = "MM Engine: C++20 limit order book";
  py::enum_<Side>(m, "Side").value("Buy", Side::Buy).value("Sell", Side::Sell);

  PYBIND11_NUMPY_DTYPE(Trade, maker, taker, price, qty, taker_side);

  py::class_<OrderBook>(m, "OrderBook")
      .def(py::init([](Price lo, Price hi, std::uint32_t max_orders, std::uint32_t max_trades) {
             return OrderBook(BookConfig{lo, hi, max_orders, max_trades});
           }),
           py::arg("min_price") = 0, py::arg("max_price") = 100000,
           py::arg("max_orders") = 1u << 20, py::arg("max_trades") = 1u << 16)
      .def("add_limit", &OrderBook::add_limit)
      .def("add_ioc", &OrderBook::add_ioc)
      .def("add_market", &OrderBook::add_market)
      .def("cancel", &OrderBook::cancel)
      .def("reduce", &OrderBook::reduce)
      .def("replace", &OrderBook::replace)
      .def("contains", &OrderBook::contains)
      .def_property_readonly("best_bid", [](const OrderBook& b) -> py::object {
        return b.best_bid() == OrderBook::kNoBid ? py::object(py::none()) : py::int_(b.best_bid());
      })
      .def_property_readonly("best_ask", [](const OrderBook& b) -> py::object {
        return b.best_ask() == OrderBook::kNoAsk ? py::object(py::none()) : py::int_(b.best_ask());
      })
      .def("depth_at", &OrderBook::depth_at)
      .def_property_readonly("order_count", &OrderBook::order_count)
      .def("snapshot", [](const OrderBook& b, Side s, int n) {
        py::array_t<Price> px(n);
        py::array_t<Qty> q(n);
        const int k = b.snapshot(s, n, px.mutable_data(), q.mutable_data());
        px.resize({k});
        q.resize({k});
        return py::make_tuple(px, q);
      })
      .def("trades", [](py::object self) {
        const auto& t = self.cast<const OrderBook&>().trades();
        // View into the engine's buffer; `self` is kept alive as the base.
        return py::array_t<Trade>({static_cast<py::ssize_t>(t.size())}, {sizeof(Trade)}, t.data(), self);
      }, "Zero-copy view of the trade tape. Copy it before calling clear_trades().")
      .def("clear_trades", &OrderBook::clear_trades);

  // ---- derivatives (mme/options.hpp)
  using namespace mme::opt;
  auto mk = [](double S, double K, double T, double r, double q, double sigma) { return Market{S, K, T, r, q, sigma}; };
  auto ty = [](bool call) { return call ? Type::Call : Type::Put; };
  auto mc_dict = [](const McResult& r) {
    py::dict d;
    d["price"] = r.price;
    d["std_error"] = r.std_error;
    d["ci95"] = py::make_tuple(r.ci_low, r.ci_high);
    d["samples"] = r.samples;
    return d;
  };
  m.def("bs_price", [=](double S, double K, double T, double r, double q, double sigma, bool call) {
    return bs_price(mk(S, K, T, r, q, sigma), ty(call));
  }, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("call") = true);
  m.def("bs_greeks", [=](double S, double K, double T, double r, double q, double sigma, bool call) {
    const Greeks g = bs_greeks(mk(S, K, T, r, q, sigma), ty(call));
    py::dict d;
    d["price"] = g.price; d["delta"] = g.delta; d["gamma"] = g.gamma;
    d["vega"] = g.vega; d["theta"] = g.theta; d["rho"] = g.rho;
    return d;
  }, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("call") = true);
  m.def("implied_vol", [=](double price, double S, double K, double T, double r, double q, bool call) {
    return implied_vol(price, mk(S, K, T, r, q, 0.0), ty(call));
  }, py::arg("price"), py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("call") = true);
  m.def("mc_price", [=](double S, double K, double T, double r, double q, double sigma, bool call, uint64_t paths,
                        int steps, bool euler, bool antithetic, bool control_variate, uint64_t seed) {
    return mc_dict(mc_price(mk(S, K, T, r, q, sigma), ty(call),
                            {paths, steps, euler ? Scheme::Euler : Scheme::Exact, antithetic, control_variate, seed}));
  }, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("call") = true,
     py::arg("paths") = 100000, py::arg("steps") = 1, py::arg("euler") = false, py::arg("antithetic") = false,
     py::arg("control_variate") = false, py::arg("seed") = 1);
  m.def("heston_price", [=](double S, double K, double T, double r, double q, double v0, double kappa, double theta,
                            double xi, double rho, bool call) {
    return heston_price(S, K, T, r, q, Heston{v0, kappa, theta, xi, rho}, ty(call));
  }, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("v0"), py::arg("kappa"),
     py::arg("theta"), py::arg("xi"), py::arg("rho"), py::arg("call") = true);
  m.def("delta_hedge", [=](double S, double K, double T, double r, double q, bool call, double mu, double sigma_real,
                           double sigma_hedge, int rebalances, uint64_t paths, uint64_t seed) {
    const HedgeResult h = delta_hedge(mk(S, K, T, r, q, sigma_hedge), ty(call),
                                      {mu, sigma_real, sigma_hedge, rebalances, paths, seed});
    py::dict d;
    d["mean_pnl"] = h.mean_pnl; d["std_pnl"] = h.std_pnl; d["mean_ito_pnl"] = h.mean_ito_pnl;
    d["corr_pnl_ito"] = h.corr_pnl_ito; d["derman_kamal_std"] = h.derman_kamal_std;
    return d;
  }, py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"), py::arg("call") = true,
     py::arg("mu") = 0.05, py::arg("sigma_real") = 0.2, py::arg("sigma_hedge") = 0.2, py::arg("rebalances") = 52,
     py::arg("paths") = 20000, py::arg("seed") = 1);
}
