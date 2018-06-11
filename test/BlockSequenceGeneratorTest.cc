#include <optional>
#include <stdexcept>
#include <tuple>

#include "BlockGenerator.h"
#include "gtest/gtest.h"

using namespace mpims;

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
