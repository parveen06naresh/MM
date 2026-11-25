// pybind11 bridge. Depth snapshots and the trade tape come back as NumPy arrays;
// the trade tape is a zero-copy view into the engine's own buffer.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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
}
