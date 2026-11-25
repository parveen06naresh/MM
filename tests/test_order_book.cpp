#include <gtest/gtest.h>

#include <random>

#include "mme/order_book.hpp"
#include "reference_book.hpp"

using namespace mme;

static BookConfig small() { return BookConfig{0, 1000, 4096, 4096}; }

TEST(OrderBook, RestsAndReportsBestPrices) {
  OrderBook b(small());
  EXPECT_EQ(b.best_bid(), OrderBook::kNoBid);
  EXPECT_EQ(b.add_limit(1, Side::Buy, 99, 10), 0);
  EXPECT_EQ(b.add_limit(2, Side::Sell, 101, 5), 0);
  EXPECT_EQ(b.best_bid(), 99);
  EXPECT_EQ(b.best_ask(), 101);
  EXPECT_EQ(b.depth_at(Side::Buy, 99), 10);
  EXPECT_EQ(b.order_count(), 2u);
}

TEST(OrderBook, PriceTimePriority) {
  OrderBook b(small());
  b.add_limit(1, Side::Sell, 100, 5);
  b.add_limit(2, Side::Sell, 100, 5);
  b.add_limit(3, Side::Sell, 99, 5);
  EXPECT_EQ(b.add_limit(10, Side::Buy, 100, 12), 12);
  const auto& t = b.trades();
  ASSERT_EQ(t.size(), 3u);
  EXPECT_EQ(t[0].maker, 3u);  // better price first
  EXPECT_EQ(t[0].price, 99);
  EXPECT_EQ(t[1].maker, 1u);  // then earlier order at the same price
  EXPECT_EQ(t[2].maker, 2u);
  EXPECT_EQ(t[2].qty, 2);
  EXPECT_EQ(b.depth_at(Side::Sell, 100), 3);
}

TEST(OrderBook, PartialFillRestsRemainder) {
  OrderBook b(small());
  b.add_limit(1, Side::Sell, 100, 4);
  EXPECT_EQ(b.add_limit(2, Side::Buy, 101, 10), 4);
  EXPECT_EQ(b.best_bid(), 101);
  EXPECT_EQ(b.depth_at(Side::Buy, 101), 6);
  EXPECT_EQ(b.best_ask(), OrderBook::kNoAsk);
}

TEST(OrderBook, MarketSweepsLevelsAndNeverRests) {
  OrderBook b(small());
  b.add_limit(1, Side::Buy, 98, 3);
  b.add_limit(2, Side::Buy, 97, 3);
  EXPECT_EQ(b.add_market(3, Side::Sell, 10), 6);
  EXPECT_EQ(b.best_bid(), OrderBook::kNoBid);
  EXPECT_EQ(b.best_ask(), OrderBook::kNoAsk);
  EXPECT_FALSE(b.contains(3));
}

TEST(OrderBook, CancelReduceReplace) {
  OrderBook b(small());
  b.add_limit(1, Side::Buy, 100, 10);
  b.add_limit(2, Side::Buy, 100, 10);
  EXPECT_TRUE(b.reduce(1, 4));        // keeps priority
  EXPECT_FALSE(b.reduce(1, 8));       // increase is not allowed in place
  b.add_market(9, Side::Sell, 4);
  EXPECT_EQ(b.trades().back().maker, 1u);
  EXPECT_FALSE(b.contains(1));
  EXPECT_EQ(b.replace(2, 101, 10), 0);  // moves to a new price
  EXPECT_EQ(b.best_bid(), 101);
  EXPECT_TRUE(b.cancel(2));
  EXPECT_FALSE(b.cancel(2));
  EXPECT_EQ(b.best_bid(), OrderBook::kNoBid);
}

TEST(OrderBook, RejectsBadInput) {
  OrderBook b(small());
  EXPECT_EQ(b.add_limit(1, Side::Buy, 2000, 1), -1);
  EXPECT_EQ(b.add_limit(1, Side::Buy, 10, 0), -1);
  EXPECT_EQ(b.add_limit(1, Side::Buy, 10, 1), 0);
  EXPECT_EQ(b.add_limit(1, Side::Buy, 11, 1), -1);  // duplicate id
}

TEST(OrderBook, BookNeverCrosses) {
  OrderBook b(small());
  std::mt19937_64 rng(7);
  for (OrderId id = 1; id < 20000; ++id) {
    const Side s = rng() & 1 ? Side::Buy : Side::Sell;
    b.add_limit(id, s, 450 + static_cast<Price>(rng() % 100), 1 + rng() % 9);
    if (b.best_bid() != OrderBook::kNoBid && b.best_ask() != OrderBook::kNoAsk) {
      ASSERT_LT(b.best_bid(), b.best_ask());
    }
    if (rng() % 3 == 0) b.cancel(1 + rng() % id);
  }
}

// Differential test: the fast engine and the simple std::map reference must
// produce the identical trade tape and book state for random message streams.
class Differential : public ::testing::TestWithParam<int> {};

TEST_P(Differential, MatchesReferenceBook) {
  const int seed = GetParam();
  OrderBook b(BookConfig{0, 200, 8192, 1 << 16});
  ref::Book r(0, 200);
  std::mt19937_64 rng(seed);
  OrderId next = 1;
  std::vector<OrderId> ids;
  for (int step = 0; step < 4000; ++step) {
    const int op = static_cast<int>(rng() % 100);
    const Side s = rng() & 1 ? Side::Buy : Side::Sell;
    const Price mid = 100;
    if (op < 55) {
      const Price px = mid + static_cast<Price>(rng() % 21) - 10;
      const Qty q = 1 + static_cast<Qty>(rng() % 20);
      ASSERT_EQ(b.add_limit(next, s, px, q), r.add_limit(next, s, px, q));
      ids.push_back(next++);
    } else if (op < 85 && !ids.empty()) {
      const OrderId id = ids[rng() % ids.size()];
      ASSERT_EQ(b.cancel(id), r.cancel(id));
    } else if (op < 93 && !ids.empty()) {
      const OrderId id = ids[rng() % ids.size()];
      const Qty nq = static_cast<Qty>(rng() % 10);
      ASSERT_EQ(b.reduce(id, nq), r.reduce(id, nq));
    } else {
      const Qty q = 1 + static_cast<Qty>(rng() % 30);
      ASSERT_EQ(b.add_market(next, s, q), r.market(next, s, q));
      ++next;
    }
    ASSERT_EQ(b.best_bid(), r.best_bid());
    ASSERT_EQ(b.best_ask(), r.best_ask());
    ASSERT_EQ(b.order_count(), r.count());
  }
  const auto& t1 = b.trades();
  ASSERT_EQ(t1.size(), r.trades.size());
  for (std::size_t i = 0; i < t1.size(); ++i) {
    ASSERT_EQ(t1[i].maker, r.trades[i].maker);
    ASSERT_EQ(t1[i].taker, r.trades[i].taker);
    ASSERT_EQ(t1[i].price, r.trades[i].price);
    ASSERT_EQ(t1[i].qty, r.trades[i].qty);
  }
  for (Price p = 0; p <= 200; ++p) {
    ASSERT_EQ(b.depth_at(Side::Buy, p), r.depth(Side::Buy, p));
    ASSERT_EQ(b.depth_at(Side::Sell, p), r.depth(Side::Sell, p));
  }
}

INSTANTIATE_TEST_SUITE_P(Seeds, Differential, ::testing::Range(1, 201));
