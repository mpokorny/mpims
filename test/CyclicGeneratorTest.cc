#include <optional>
#include <stdexcept>
#include <tuple>

#include "BlockGenerator.h"
#include "gtest/gtest.h"

using namespace mpims;

TEST(CyclicGenerator, GroupIndexLimit) {

  const std::size_t group_size = 2, block_length = 2, axis_length = 4;

  EXPECT_NO_THROW(
    CyclicGenerator::initial_state(block_length, group_size, axis_length, 0));
  EXPECT_NO_THROW(
    CyclicGenerator::initial_state(block_length, group_size, axis_length, 1));
  EXPECT_THROW(
    CyclicGenerator::initial_state(block_length, group_size, axis_length, 2),
    std::domain_error);
}

TEST(CyclicGenerator, BoundedCycle) {

  const std::size_t group_size = 2, block_length = 2, axis_length = 4;

  auto s0 =
    CyclicGenerator::initial_state(block_length, group_size, axis_length, 0);
  CyclicGenerator::State s1;
  std::tie(s1, std::ignore) = CyclicGenerator::apply(s0);
  CyclicGenerator::State s2;
  std::optional<mpims::block_t> end;
  std::tie(s2, end) = CyclicGenerator::apply(s1);
  EXPECT_FALSE(end);

  s0 = CyclicGenerator::initial_state(block_length, group_size, axis_length, 1);
  std::tie(s1, std::ignore) = CyclicGenerator::apply(s0);
  std::tie(s2, end) = CyclicGenerator::apply(s1);
  EXPECT_FALSE(end);
}

TEST(CyclicGenerator, BlockSequence) {

  const std::size_t group_size = 2, block_length = 2, axis_length = 8;

  for (std::size_t gi = 0; gi < group_size; ++gi) {
    auto st =
      CyclicGenerator::initial_state(block_length, group_size, axis_length, gi);
    std::optional<mpims::block_t> blk;
    std::optional<std::size_t> b_prev;
    std::size_t b0;
    do {
      std::tie(st, blk) = CyclicGenerator::apply(st);
      if (blk) {
        b0 = std::get<0>(blk.value());
        if (b_prev)
          EXPECT_EQ(b0, b_prev.value() + group_size * block_length);
        b_prev = b0;
      }
    } while (blk);
  }

}

TEST(CyclicGenerator, NonDivisibleSequence) {

  const std::size_t group_size = 2, block_length = 3, axis_length = 13;

  std::optional<mpims::block_t> blk;
  auto st =
    CyclicGenerator::initial_state(block_length, group_size, axis_length, 0);
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(0, block_length)); // (0, 3)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(6, block_length)); // (6, 3)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(12, 1));           // (12, 1)
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_FALSE(blk);

  st = CyclicGenerator::initial_state(block_length, group_size, axis_length, 1);
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
  auto st =
    CyclicGenerator::initial_state(block_length, group_size, axis_length, 0);
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_EQ(blk, std::make_tuple(0, block_length));
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_FALSE(blk);

  st = CyclicGenerator::initial_state(block_length, group_size, axis_length, 1);
  std::tie(st, blk) = CyclicGenerator::apply(st);
  EXPECT_FALSE(blk);
}

TEST(CyclicGenerator, ApproxUnboundedCycle) {

  const std::size_t group_size = 2, block_length = 2;

  for (std::size_t gi = 0; gi < group_size; ++gi) {
    auto st = CyclicGenerator::initial_state(2, 2, std::nullopt, gi);
    std::optional<mpims::block_t> blk;
    std::optional<std::size_t> b_prev;
    do {
      std::tie(st, blk) = CyclicGenerator::apply(st);
      EXPECT_TRUE(blk);

      // prevent infinite loop
      if (b_prev) {
        std::size_t b0 = std::get<0>(blk.value());
        ASSERT_EQ(b0, b_prev.value() + group_size * block_length);
        b_prev = b0;
      }

    } while (std::get<0>(blk.value()) < 10000000);
  }
}

int
main(int argc, char *argv[]) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}
