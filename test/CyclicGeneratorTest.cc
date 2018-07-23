#include <optional>
#include <stdexcept>
#include <tuple>

#include "BlockGenerator.h"
#include "gtest/gtest.h"

using namespace mpims;

TEST(CyclicGenerator, RankLimit) {

  const std::size_t group_size = 2, block_length = 2, axis_length = 4;

  auto states =
    CyclicGenerator::initial_states(block_length, group_size, axis_length);

  EXPECT_NO_THROW(states(0));
  EXPECT_NO_THROW(states(1));
  EXPECT_THROW(states(2), std::domain_error);
}

TEST(CyclicGenerator, BoundedCycle) {

  const std::size_t group_size = 2, block_length = 2, axis_length = 4;

  auto states =
    CyclicGenerator::initial_states(block_length, group_size, axis_length);
  auto s0 = states(0) ;
  CyclicGenerator::State s1;
  std::tie(s1, std::ignore) = CyclicGenerator::apply(s0);
  CyclicGenerator::State s2;
  std::optional<mpims::block_t> end;
  std::tie(s2, end) = CyclicGenerator::apply(s1);
  EXPECT_FALSE(end);

  s0 = states(1);
  std::tie(s1, std::ignore) = CyclicGenerator::apply(s0);
  std::tie(s2, end) = CyclicGenerator::apply(s1);
  EXPECT_FALSE(end);
}

TEST(CyclicGenerator, BlockSequence) {

  const std::size_t group_size = 2, block_length = 2, axis_length = 8;
  std::size_t step = group_size * block_length;

  auto states =
    CyclicGenerator::initial_states(block_length, group_size, axis_length);

  for (std::size_t rank = 0; rank < group_size; ++rank) {
    auto st = states(rank);
    std::optional<mpims::block_t> blk;
    std::optional<std::size_t> b_prev;
    std::size_t b0;
    std::tie(st, blk) = CyclicGenerator::apply(st);
    while (blk) {
      b0 = std::get<0>(blk.value());
      EXPECT_EQ(b0, b_prev.value_or(b0 - step) + step);
      b_prev = b0;
      std::tie(st, blk) = CyclicGenerator::apply(st);
    };
  }

}

TEST(CyclicGenerator, NonDivisibleSequence) {

  const std::size_t group_size = 2, block_length = 3, axis_length = 13;

  std::optional<mpims::block_t> blk;
  auto states =
    CyclicGenerator::initial_states(block_length, group_size, axis_length);

  auto st = states(0);
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(0, block_length)); // (0, 3)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(6, block_length)); // (6, 3)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(12, 1));           // (12, 1)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_FALSE(blk);

  st = states(1);
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(3, block_length)); // (3, 3)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(9, block_length)); // (9, 3)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_FALSE(blk);
}

TEST(CyclicGenerator, EmptyGroup) {
  const std::size_t group_size = 2, block_length = 4, axis_length = 4;

  std::optional<mpims::block_t> blk;
  auto states =
    CyclicGenerator::initial_states(block_length, group_size, axis_length);

  auto st = states(0);
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(0, block_length));
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_FALSE(blk);

  st = states(1);
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_FALSE(blk);
}

TEST(CyclicGenerator, ApproxUnboundedCycle) {

  const std::size_t group_size = 2, block_length = 2;
  std::size_t step = group_size * block_length;
  auto states =
    CyclicGenerator::initial_states(block_length, group_size, std::nullopt);

  for (std::size_t rank = 0; rank < group_size; ++rank) {
    auto st = states(rank);
    std::optional<mpims::block_t> blk;
    std::optional<std::size_t> b_prev;
    std::tie(st, blk) = CyclicGenerator::apply(st);
    ASSERT_TRUE(blk);
    while (std::get<0>(blk.value()) < 10000000) {
      // prevent infinite loop in this test (block origin not advancing is
      // tested separately, so this assertion is a sort of backstop)
      std::size_t b0 = std::get<0>(blk.value());
      ASSERT_EQ(b0, b_prev.value_or(b0 - step) + step);
      b_prev = b0;

      std::tie(st, blk) = CyclicGenerator::apply(st);
      ASSERT_TRUE(blk);
    };
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
