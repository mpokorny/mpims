#include <optional>
#include <stdexcept>
#include <tuple>

#include "BlockGenerator.h"
#include "gtest/gtest.h"

using namespace mpims;

TEST(BlockSequenceGenerator, BlocksOverlap) {
  const std::vector<block_t> good0{ block_t(0, 2), block_t(5, 3) };
  const std::vector<block_t> good1{ block_t(0, 2), block_t(2, 3) };
  const std::vector<block_t> bad{ block_t(0, 2), block_t(1, 3) };

  EXPECT_NO_THROW(BlockSequenceGenerator::initial_state(good0, std::nullopt));
  EXPECT_NO_THROW(BlockSequenceGenerator::initial_state(good1, std::nullopt));
  EXPECT_THROW(
    BlockSequenceGenerator::initial_state(bad, std::nullopt),
    std::domain_error);
}

TEST(BlockSequenceGenerator, SimpleSequence) {

  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1)};

  auto st = BlockSequenceGenerator::initial_state(blocks, 13);
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

TEST(BlockSequenceGenerator, TruncatedSequence) {

  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1)};

  auto st = BlockSequenceGenerator::initial_state(blocks, 7);
  std::optional<block_t> blk;
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  EXPECT_TRUE(blk);
  if (blk) {
    EXPECT_EQ(blk.value(), blocks[0]);
  }
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  EXPECT_TRUE(blk);
  if (blk) {
    EXPECT_EQ(blk.value(), block_t(5, 2));
  }
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  EXPECT_FALSE(blk);
}

TEST(BlockSequenceGenerator, SimpleSequenceUnboundedAxis) {

  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1)};

  auto st = BlockSequenceGenerator::initial_state(blocks, std::nullopt);
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

TEST(BlockSequenceGenerator, RepeatingSequenceBoundedAxis) {

  const std::size_t rep = 15;
  const std::size_t axreps = 3;
  const std::vector<block_t> blocks{
    block_t(0, 2),
      block_t(5, 3),
      block_t(12, 1),
      block_t(rep, 0)};

  auto st = BlockSequenceGenerator::initial_state(blocks, axreps * rep);
  std::optional<block_t> blk;
  for (std::size_t n = 0; n < axreps; ++n)
    for (std::size_t i = 0; i < blocks.size() - 1; ++i) {
      std::tie(st, blk) = BlockSequenceGenerator::apply(st);
      EXPECT_TRUE(blk);
      if (blk) {
        auto rblk =
          std::make_tuple(
            std::get<0>(blocks[i]) + n * rep,
            std::get<1>(blocks[i]));
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

  auto st = BlockSequenceGenerator::initial_state(blocks, len);
  std::optional<block_t> blk;
  for (std::size_t n = 0; n < axreps - 1; ++n)
    for (std::size_t i = 0; i < blocks.size() - 1; ++i) {
      std::tie(st, blk) = BlockSequenceGenerator::apply(st);
      ASSERT_TRUE(blk);
      block_t rblk =
        std::make_tuple(
          std::get<0>(blocks[i]) + n * rep,
          std::get<1>(blocks[i]));
      EXPECT_EQ(blk.value(), rblk);
    }
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  ASSERT_TRUE(blk);
  block_t rblk =
    std::make_tuple(
      std::get<0>(blocks[0]) + (axreps - 1) * rep,
      std::get<1>(blocks[0]));
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

  auto st = BlockSequenceGenerator::initial_state(blocks, std::nullopt);
  std::optional<block_t> blk;
  std::tie(st, blk) = BlockSequenceGenerator::apply(st);
  ASSERT_TRUE(blk);
  std::size_t n = 0, bi = 0;
  while (std::get<0>(blk.value()) < 10000000) {

    block_t rblk =
      std::make_tuple(
        std::get<0>(blocks[bi]) + n * rep,
        std::get<1>(blocks[bi]));
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
