// MM Engine: price-time priority limit order book with a zero-allocation hot path.
//
// Design
//  * Prices are integer ticks inside a fixed band [min_price, max_price], so each
//    price level is an O(1) array lookup (no tree, no hashing on the price path).
//  * Resting orders live in a preallocated node pool and form an intrusive,
//    doubly linked FIFO per level. Add, fill and cancel never touch the heap.
//  * Order id -> node lookup is an open-addressing hash table (linear probing,
//    backward-shift deletion), also preallocated.
//  * Trades go to a tape that is reserved up front; the caller drains it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mme {

using Price = std::int64_t;  // integer ticks
using Qty = std::int64_t;
using OrderId = std::uint64_t;

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

struct Trade {
  OrderId maker;
  OrderId taker;
  Price price;
  Qty qty;
  std::uint8_t taker_side;  // 0 = buy aggressor, 1 = sell aggressor
};

struct BookConfig {
  Price min_price = 0;
  Price max_price = 100'000;
  std::uint32_t max_orders = 1u << 20;
  std::uint32_t max_trades = 1u << 16;
};

class OrderBook {
 public:
  static constexpr Price kNoBid = std::numeric_limits<Price>::min();
  static constexpr Price kNoAsk = std::numeric_limits<Price>::max();

  explicit OrderBook(const BookConfig& cfg = {});

  // Limit order: matches against the opposite side up to `px`, rests the rest.
  // Returns the quantity filled immediately. Rejects (returns -1) a duplicate
  // id, a price outside the band, a non-positive quantity, or a full pool.
  Qty add_limit(OrderId id, Side side, Price px, Qty qty);
  // Immediate-or-cancel: matches up to `px`, never rests.
  Qty add_ioc(OrderId id, Side side, Price px, Qty qty);
  // Market order: sweeps the opposite side, never rests.
  Qty add_market(OrderId id, Side side, Qty qty);

  bool cancel(OrderId id);
  // Shrinks a resting order in place and keeps its queue position.
  bool reduce(OrderId id, Qty new_qty);
  // Cancel and re-add: loses queue position (exchange semantics). Returns the
  // filled quantity of the new order, or -1 if `id` was not resting.
  Qty replace(OrderId id, Price px, Qty qty);

  Price best_bid() const { return best_bid_; }
  Price best_ask() const { return best_ask_; }
  Qty depth_at(Side side, Price px) const;
  std::size_t order_count() const { return live_orders_; }
  bool contains(OrderId id) const { return find_slot(id) != kEmpty; }

  // Writes up to n levels (best first). Returns the number written.
  int snapshot(Side side, int n, Price* px_out, Qty* qty_out) const;

  const std::vector<Trade>& trades() const { return trades_; }
  void clear_trades() { trades_.clear(); }

  Price min_price() const { return min_px_; }
  Price max_price() const { return max_px_; }

 private:
  static constexpr std::uint32_t kNil = std::numeric_limits<std::uint32_t>::max();
  static constexpr std::uint32_t kEmpty = std::numeric_limits<std::uint32_t>::max();

  struct Node {
    OrderId id;
    Qty qty;
    Price px;
    std::uint32_t prev;
    std::uint32_t next;
    Side side;
  };
  struct Level {
    Qty total = 0;
    std::uint32_t head = kNil;
    std::uint32_t tail = kNil;
  };
  struct Slot {
    OrderId key;
    std::uint32_t node;  // kEmpty when the slot is free
  };

  Qty match(OrderId taker, Side side, Price limit, Qty qty);
  void rest(OrderId id, Side side, Price px, Qty qty);
  void unlink(std::uint32_t n);
  void refresh_best_bid(Price from);
  void refresh_best_ask(Price from);
  Level& level(Side s, Price px) {
    return (s == Side::Buy ? bids_ : asks_)[static_cast<std::size_t>(px - min_px_)];
  }
  const Level& level(Side s, Price px) const {
    return (s == Side::Buy ? bids_ : asks_)[static_cast<std::size_t>(px - min_px_)];
  }

  // id map
  std::uint32_t find_slot(OrderId id) const;
  bool map_insert(OrderId id, std::uint32_t node);
  void map_erase(OrderId id);
  std::size_t hash(OrderId id) const {
    std::uint64_t x = id * 0x9E3779B97F4A7C15ull;
    return static_cast<std::size_t>(x >> shift_);
  }

  Price min_px_, max_px_;
  std::vector<Level> bids_, asks_;
  std::vector<Node> pool_;
  std::vector<std::uint32_t> free_;
  std::vector<Slot> map_;
  unsigned shift_;
  std::size_t mask_;
  std::vector<Trade> trades_;
  Price best_bid_ = kNoBid, best_ask_ = kNoAsk;
  std::size_t live_orders_ = 0, bid_orders_ = 0, ask_orders_ = 0;
};

}  // namespace mme
