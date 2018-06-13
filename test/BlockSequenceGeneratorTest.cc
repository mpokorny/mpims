#include <optional>
#include <stdexcept>
#include <tuple>

#include "BlockGenerator.h"
#include "gtest/gtest.h"

using namespace mpims;

TEST(BlockSequenceGenerator, BlocksOverlap) {
  const std::vector<block_t> good0{ block_t(0, 2), block_t(5, 3) };
  const std::vector<block_t> good1{ block_t(2, 2), block_t(8, 3) };
  const std::vector<block_t> bad0{ block_t(4, 2), block_t(5, 3) };
  const std::vector<block_t> bad1{ block_t(4, std::nullopt), block_t(5, 3) };

  auto states =
    BlockSequenceGenerator::initial_states(
      std::vector{good0, good1, bad0, bad1},
      std::nullopt);

  EXPECT_NO_THROW(states(0));
  EXPECT_NO_THROW(states(1));
  EXPECT_THROW(states(2), std::domain_error);
  EXPECT_THROW(states(3), std::domain_error);
}

TEST(BlockSequenceGenerator, SimpleSequence) {

  const std::vector<std::vector<block_t> > all_blocks{
    std::vector{block_t(0, 2), block_t(5, 3), block_t(12, 1)},
      std::vector{block_t(2, 2), block_t(8, 3), block_t(13, 1)}};

  auto states = BlockSequenceGenerator::initial_states(all_blocks, 14);
  for (std::size_t rank = 0; rank < all_blocks.size(); ++rank) {
    auto blocks = all_blocks[rank];
    auto st = states(rank);
    std::optional<block_t> blk;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      std::tie(st, blk) = BlockSequenceGenerator::apply(st);
      EXPECT_TRUE(blk);
      if (blk) {
        EXPECT_EQ(blk.value(), blocks[i]);
      }
    }
    std::tie(st, blk) = BlockSequenceGenerator::apply(st);
    EXPECT_FALSE(blk);
  }
}

TEST(BlockSequenceGenerator, TruncatedSequence) {

  const std::vector<std::vector<block_t> > all_blocks{
    std::vector{block_t(0, 2), block_t(5, 3), block_t(12, 1)},
      std::vector{block_t(2, 2), block_t(8, 3), block_t(13, 1)}};
  const std::size_t axis_length = 7;

  auto states = BlockSequenceGenerator::initial_states(all_blocks, axis_length);
  for (std::size_t rank = 0; rank < all_blocks.size(); ++rank) {
    auto blocks = all_blocks[rank];
    auto st = states(rank);
    std::optional<block_t> blk;
    std::tie(st, blk) = BlockSequenceGenerator::apply(st);
    EXPECT_TRUE(blk);
    if (blk) {
      EXPECT_EQ(blk.value(), blocks[0]);
    }
    std::tie(st, blk) = BlockSequenceGenerator::apply(st);
    EXPECT_EQ(blk.has_value(), std::get<0>(blocks[1]) <= axis_length);
    if (blk) {
      std::size_t b1;
      std::optional<std::size_t> b1len;
      std::tie(b1, b1len) = blocks[1];
      EXPECT_EQ(
        blk.value(),
        block_t(b1, std::min(b1 + b1len.value(), axis_length) - b1));
    }
    std::tie(st, blk) = BlockSequenceGenerator::apply(st);
    EXPECT_FALSE(blk);
  }
}

TEST(BlockSequenceGenerator, SimpleSequenceUnboundedAxis) {

  const std::vector<std::vector<block_t> > all_blocks{
    std::vector{block_t(0, 2), block_t(5, 3), block_t(12, 1)},
      std::vector{block_t(2, 2), block_t(8, 3), block_t(13, 1)}};

  auto states =
    BlockSequenceGenerator::initial_states(all_blocks, std::nullopt);
  for (std::size_t rank = 0; rank < all_blocks.size(); ++rank) {
    auto blocks = all_blocks[rank];
    auto st = states(rank);
    std::optional<block_t> blk;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      std::tie(st, blk) = BlockSequenceGenerator::apply(st);
      EXPECT_TRUE(blk);
      if (blk) {
        EXPECT_EQ(blk.value(), blocks[i]);
      }
    }
    std::tie(st, blk) = BlockSequenceGenerator::apply(st);
    EXPECT_FALSE(blk);
  }
}

TEST(BlockSequenceGenerator, RepeatingSequenceBoundedAxis) {

  const std::size_t rep = 15;
  const std::size_t axreps = 3;
  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1),
      block_t(rep, 0)};

  auto st =
    BlockSequenceGenerator::initial_states(
      std::vector{blocks},
      axreps * rep)(0);
  std::optional<block_t> blk;
  for (std::size_t n = 0; n < axreps; ++n)
    for (std::size_t i = 0; i < blocks.size() - 1; ++i) {
      std::tie(st, blk) = BlockSequenceGenerator::apply(st);
      EXPECT_TRUE(blk);
      if (blk) {
        auto rblk =
          std::make_tuple(
            std::get<0>(blocks[i]) + n * rep,
            std::get<1>(blocks[i]).value());
        EXPECT_EQ(blk.value(), rblk);
      }
    }
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  EXPECT_FALSE(blk);
}

TEST(BlockSequenceGenerator, TruncatedRepeatingSequence) {

  const std::size_t rep = 15;
  const std::size_t axreps = 3;
  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1),
      block_t(rep, 0)};
  const std::size_t len = axreps * rep - 9;

  auto st = BlockSequenceGenerator::initial_states(std::vector{blocks}, len)(0);
  std::optional<block_t> blk;
  for (std::size_t n = 0; n < axreps - 1; ++n)
    for (std::size_t i = 0; i < blocks.size() - 1; ++i) {
      std::tie(st, blk) = BlockSequenceGenerator::apply(st);
      ASSERT_TRUE(blk);
      block_t rblk =
        std::make_tuple(
          std::get<0>(blocks[i]) + n * rep,
          std::get<1>(blocks[i]).value());
      EXPECT_EQ(blk.value(), rblk);
    }
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  ASSERT_TRUE(blk);
  block_t rblk =
    std::make_tuple(
      std::get<0>(blocks[0]) + (axreps - 1) * rep,
      std::get<1>(blocks[0]).value());
  EXPECT_EQ(blk.value(), rblk);
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  ASSERT_TRUE(blk);
  std::size_t b0 = std::get<0>(blocks[1]) + (axreps - 1) * rep;
  rblk = std::make_tuple(b0, len - b0);
  EXPECT_EQ(blk.value(), rblk);

  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  EXPECT_FALSE(blk);
}

TEST(BlockSequenceGenerator, ApproxRepeatingSequenceUnbounded) {
  const std::size_t rep = 15;
  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1),
      block_t(rep, 0)};

  auto st =
    BlockSequenceGenerator::initial_states(
      std::vector{blocks},
      std::nullopt)(0);
  std::optional<block_t> blk;
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  ASSERT_TRUE(blk);
  std::size_t n = 0, bi = 0;
  while (std::get<0>(blk.value()) < 10000000) {

    block_t rblk =
      std::make_tuple(
        std::get<0>(blocks[bi]) + n * rep,
        std::get<1>(blocks[bi]).value());
    ASSERT_EQ(blk.value(), rblk);

    ++bi;
    if (bi == blocks.size() - 1) {
      bi = 0;
      ++n;
    }

    std::tie(st, blk) = BlockSequenceGenerator::apply(st);
    ASSERT_TRUE(blk);
  }
}

TEST(BlockSequenceGenerator, RepeatingSequenceExcess) {

  const std::size_t rep = 15;
  const std::size_t axreps = 3;
  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1),
      block_t(rep, 0),
      block_t(rep + 2, 2)};

  auto st =
    BlockSequenceGenerator::initial_states(
      std::vector{blocks},
      axreps * rep)(0);
  std::optional<block_t> blk;
  for (std::size_t n = 0; n < axreps; ++n)
    for (std::size_t i = 0; i < blocks.size() - 2; ++i) {
      std::tie(st, blk) = BlockSequenceGenerator::apply(st);
      EXPECT_TRUE(blk);
      if (blk) {
        auto rblk =
          std::make_tuple(
            std::get<0>(blocks[i]) + n * rep,
            std::get<1>(blocks[i]).value());
        EXPECT_EQ(blk.value(), rblk);
      }
    }
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  EXPECT_FALSE(blk);
}

int
main(int argc, char *argv[]) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
