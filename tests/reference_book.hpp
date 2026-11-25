// A deliberately simple reference book (std::map + std::deque) used only to
// cross-check the fast engine in randomized differential tests.
#pragma once
#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

#include "mme/order_book.hpp"

namespace ref {
using namespace mme;
struct O { OrderId id; Qty qty; };
class Book {
 public:
  Book(Price lo, Price hi) : lo_(lo), hi_(hi) {}
  Qty add_limit(OrderId id, Side s, Price px, Qty q, bool rest_rest = true) {
    if (q <= 0 || px < lo_ || px > hi_) return -1;
    if (rest_rest && where_.count(id)) return -1;
    Qty filled = 0;
    while (q > 0) {
      if (s == Side::Buy) {
        if (asks_.empty() || asks_.begin()->first > px) break;
        filled += sweep(asks_.begin(), id, s, q, asks_);
      } else {
        if (bids_.empty() || bids_.rbegin()->first < px) break;
        auto it = std::prev(bids_.end());
        filled += sweep(it, id, s, q, bids_);
      }
    }
    if (q > 0 && rest_rest) {
      (s == Side::Buy ? bids_ : asks_)[px].push_back({id, q});
      where_[id] = {s, px};
    }
    return filled;
  }
  Qty market(OrderId id, Side s, Qty q) { return add_limit(id, s, s == Side::Buy ? hi_ : lo_, q, false); }
  bool cancel(OrderId id) {
    auto w = where_.find(id);
    if (w == where_.end()) return false;
    auto& side = w->second.first == Side::Buy ? bids_ : asks_;
    auto& dq = side[w->second.second];
    for (auto it = dq.begin(); it != dq.end(); ++it)
      if (it->id == id) { dq.erase(it); break; }
    if (dq.empty()) side.erase(w->second.second);
    where_.erase(w);
    return true;
  }
  bool reduce(OrderId id, Qty nq) {
    auto w = where_.find(id);
    if (w == where_.end()) return false;
    if (nq <= 0) return cancel(id);
    auto& dq = (w->second.first == Side::Buy ? bids_ : asks_)[w->second.second];
    for (auto& o : dq)
      if (o.id == id) { if (nq >= o.qty) return false; o.qty = nq; return true; }
    return false;
  }
  Price best_bid() const { return bids_.empty() ? OrderBook::kNoBid : bids_.rbegin()->first; }
  Price best_ask() const { return asks_.empty() ? OrderBook::kNoAsk : asks_.begin()->first; }
  Qty depth(Side s, Price px) const {
    const auto& side = s == Side::Buy ? bids_ : asks_;
    auto it = side.find(px);
    Qty t = 0;
    if (it != side.end()) for (auto& o : it->second) t += o.qty;
    return t;
  }
  std::size_t count() const { return where_.size(); }
  std::vector<Trade> trades;

 private:
  template <class It, class M>
  Qty sweep(It it, OrderId taker, Side s, Qty& q, M& side) {
    Qty f = 0;
    auto& dq = it->second;
    const Price px = it->first;
    while (q > 0 && !dq.empty()) {
      O& m = dq.front();
      Qty x = std::min(m.qty, q);
      trades.push_back({m.id, taker, px, x, static_cast<std::uint8_t>(s)});
      m.qty -= x; q -= x; f += x;
      if (m.qty == 0) { where_.erase(m.id); dq.pop_front(); }
    }
    if (dq.empty()) side.erase(px);
    return f;
  }
  Price lo_, hi_;
  std::map<Price, std::deque<O>> bids_, asks_;
  std::unordered_map<OrderId, std::pair<Side, Price>> where_;
};
}  // namespace ref
