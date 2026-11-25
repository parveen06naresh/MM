#include "mme/order_book.hpp"

#include <bit>
#include <stdexcept>

namespace mme {

OrderBook::OrderBook(const BookConfig& cfg)
    : min_px_(cfg.min_price), max_px_(cfg.max_price) {
  if (cfg.max_price < cfg.min_price) throw std::invalid_argument("empty price band");
  const auto levels = static_cast<std::size_t>(cfg.max_price - cfg.min_price + 1);
  bids_.assign(levels, Level{});
  asks_.assign(levels, Level{});
  pool_.resize(cfg.max_orders);
  free_.reserve(cfg.max_orders);
  for (std::uint32_t i = cfg.max_orders; i-- > 0;) free_.push_back(i);
  // Hash table at <= 50% load keeps probe chains short.
  const std::size_t cap = std::bit_ceil(static_cast<std::size_t>(cfg.max_orders) * 2);
  map_.assign(cap, Slot{0, kEmpty});
  mask_ = cap - 1;
  shift_ = 64u - static_cast<unsigned>(std::countr_zero(cap));
  trades_.reserve(cfg.max_trades);
}

// ---------------------------------------------------------------- id map
std::uint32_t OrderBook::find_slot(OrderId id) const {
  for (std::size_t i = hash(id);; i = (i + 1) & mask_) {
    const Slot& s = map_[i];
    if (s.node == kEmpty) return kEmpty;
    if (s.key == id) return static_cast<std::uint32_t>(i);
  }
}

bool OrderBook::map_insert(OrderId id, std::uint32_t node) {
  for (std::size_t i = hash(id);; i = (i + 1) & mask_) {
    Slot& s = map_[i];
    if (s.node == kEmpty) {
      s = Slot{id, node};
      return true;
    }
    if (s.key == id) return false;
  }
}

// Linear probing with backward-shift deletion: no tombstones, so lookups stay
// fast no matter how many orders have been cancelled.
void OrderBook::map_erase(OrderId id) {
  std::size_t i = find_slot(id);
  if (i == kEmpty) return;
  for (std::size_t j = (i + 1) & mask_;; j = (j + 1) & mask_) {
    if (map_[j].node == kEmpty) break;
    const std::size_t home = hash(map_[j].key);
    // Move j back into the hole at i if its home position is not in (i, j].
    const bool in_range = (i <= j) ? (i < home && home <= j) : (i < home || home <= j);
    if (!in_range) {
      map_[i] = map_[j];
      i = j;
    }
  }
  map_[i].node = kEmpty;
}

// ---------------------------------------------------------------- book
void OrderBook::refresh_best_bid(Price from) {
  if (bid_orders_ == 0) {
    best_bid_ = kNoBid;
    return;
  }
  for (Price p = from; p >= min_px_; --p)
    if (bids_[static_cast<std::size_t>(p - min_px_)].head != kNil) {
      best_bid_ = p;
      return;
    }
  best_bid_ = kNoBid;
}

void OrderBook::refresh_best_ask(Price from) {
  if (ask_orders_ == 0) {
    best_ask_ = kNoAsk;
    return;
  }
  for (Price p = from; p <= max_px_; ++p)
    if (asks_[static_cast<std::size_t>(p - min_px_)].head != kNil) {
      best_ask_ = p;
      return;
    }
  best_ask_ = kNoAsk;
}

void OrderBook::unlink(std::uint32_t n) {
  Node& o = pool_[n];
  Level& lv = level(o.side, o.px);
  if (o.prev != kNil) pool_[o.prev].next = o.next; else lv.head = o.next;
  if (o.next != kNil) pool_[o.next].prev = o.prev; else lv.tail = o.prev;
  lv.total -= o.qty;
  map_erase(o.id);
  free_.push_back(n);
  --live_orders_;
  if (o.side == Side::Buy) {
    --bid_orders_;
    if (lv.head == kNil && o.px == best_bid_) refresh_best_bid(o.px - 1);
  } else {
    --ask_orders_;
    if (lv.head == kNil && o.px == best_ask_) refresh_best_ask(o.px + 1);
  }
}

Qty OrderBook::match(OrderId taker, Side side, Price limit, Qty qty) {
  Qty filled = 0;
  const auto tside = static_cast<std::uint8_t>(side);
  while (qty > 0) {
    Price px;
    if (side == Side::Buy) {
      px = best_ask_;
      if (px == kNoAsk || px > limit) break;
    } else {
      px = best_bid_;
      if (px == kNoBid || px < limit) break;
    }
    const Side maker_side = side == Side::Buy ? Side::Sell : Side::Buy;
    Level& lv = level(maker_side, px);
    while (qty > 0 && lv.head != kNil) {
      const std::uint32_t n = lv.head;
      Node& m = pool_[n];
      const Qty q = m.qty < qty ? m.qty : qty;
      trades_.push_back(Trade{m.id, taker, px, q, tside});
      qty -= q;
      filled += q;
      if (q == m.qty) {
        unlink(n);  // also advances best price when the level empties
      } else {
        m.qty -= q;
        lv.total -= q;
      }
    }
  }
  return filled;
}

void OrderBook::rest(OrderId id, Side side, Price px, Qty qty) {
  const std::uint32_t n = free_.back();
  free_.pop_back();
  map_insert(id, n);
  Level& lv = level(side, px);
  pool_[n] = Node{id, qty, px, lv.tail, kNil, side};
  if (lv.tail != kNil) pool_[lv.tail].next = n; else lv.head = n;
  lv.tail = n;
  lv.total += qty;
  ++live_orders_;
  if (side == Side::Buy) {
    ++bid_orders_;
    if (best_bid_ == kNoBid || px > best_bid_) best_bid_ = px;
  } else {
    ++ask_orders_;
    if (best_ask_ == kNoAsk || px < best_ask_) best_ask_ = px;
  }
}

Qty OrderBook::add_limit(OrderId id, Side side, Price px, Qty qty) {
  if (qty <= 0 || px < min_px_ || px > max_px_ || find_slot(id) != kEmpty) return -1;
  const Qty filled = match(id, side, px, qty);
  if (filled < qty) {
    if (free_.empty()) return -1;
    rest(id, side, px, qty - filled);
  }
  return filled;
}

Qty OrderBook::add_ioc(OrderId id, Side side, Price px, Qty qty) {
  if (qty <= 0 || px < min_px_ || px > max_px_) return -1;
  return match(id, side, px, qty);
}

Qty OrderBook::add_market(OrderId id, Side side, Qty qty) {
  if (qty <= 0) return -1;
  return match(id, side, side == Side::Buy ? max_px_ : min_px_, qty);
}

bool OrderBook::cancel(OrderId id) {
  const std::uint32_t s = find_slot(id);
  if (s == kEmpty) return false;
  unlink(map_[s].node);
  return true;
}

bool OrderBook::reduce(OrderId id, Qty new_qty) {
  const std::uint32_t s = find_slot(id);
  if (s == kEmpty) return false;
  Node& o = pool_[map_[s].node];
  if (new_qty <= 0) {
    unlink(map_[s].node);
    return true;
  }
  if (new_qty >= o.qty) return false;  // increases must re-queue via replace()
  level(o.side, o.px).total -= o.qty - new_qty;
  o.qty = new_qty;
  return true;
}

Qty OrderBook::replace(OrderId id, Price px, Qty qty) {
  const std::uint32_t s = find_slot(id);
  if (s == kEmpty) return -1;
  const Side side = pool_[map_[s].node].side;
  unlink(map_[s].node);
  return add_limit(id, side, px, qty);
}

Qty OrderBook::depth_at(Side side, Price px) const {
  if (px < min_px_ || px > max_px_) return 0;
  return level(side, px).total;
}

int OrderBook::snapshot(Side side, int n, Price* px_out, Qty* qty_out) const {
  int k = 0;
  if (side == Side::Buy) {
    for (Price p = best_bid_; best_bid_ != kNoBid && p >= min_px_ && k < n; --p) {
      const Level& lv = bids_[static_cast<std::size_t>(p - min_px_)];
      if (lv.head != kNil) { px_out[k] = p; qty_out[k] = lv.total; ++k; }
    }
  } else {
    for (Price p = best_ask_; best_ask_ != kNoAsk && p <= max_px_ && k < n; ++p) {
      const Level& lv = asks_[static_cast<std::size_t>(p - min_px_)];
      if (lv.head != kNil) { px_out[k] = p; qty_out[k] = lv.total; ++k; }
    }
  }
  return k;
}

}  // namespace mme
