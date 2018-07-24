#include <optional>
#include <unordered_map>
#include <vector>

#include "BlockGenerator.h"
#include "gtest/gtest.h"

using namespace mpims;

bool
are_disjoint(const finite_block_t& x, const finite_block_t& y) {
  std::size_t x0, xlen, y0, ylen;
  std::tie(x0, xlen) = x;
  std::tie(y0, ylen) = y;
  return x0 + xlen <= y0 || y0 + ylen <= x0;
}

TEST(QuasiEquipartitionGenerator, Blocks) {

  std::size_t axis_length = 25;
  std::size_t orders[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (auto& order : orders) {
    auto sts = QuasiEquipartitionGenerator::initial_states(order, axis_length);
    // collect blocks for all ranks
    std::vector<finite_block_t> blocks;
    for (std::size_t rank = 0; rank < order; ++rank) {
      auto st = sts(rank);
      std::optional<block_t> oblk;
      std::tie(st, oblk) = QuasiEquipartitionGenerator::apply(st);
      EXPECT_FALSE(st);
      ASSERT_TRUE(oblk);
      std::size_t b0;
      std::optional<std::size_t> blen;
      std::tie(b0, blen) = oblk.value();
      ASSERT_TRUE(blen);
      blocks.emplace_back(finite_block_t{b0, blen.value()});
    }
    // check for disjoint blocks, find min block offset, and compute block size
    // distribution
    std::size_t minb = std::get<0>(blocks[0]);
    std::unordered_map<std::size_t, std::size_t> bsizes;
    for (std::size_t i = 0; i < order; ++i) {
      std::size_t i0, ilen;
      std::tie(i0, ilen) = blocks[i];
      minb = std::min(minb, i0);
      if (bsizes.count(ilen) == 0)
        bsizes[ilen] = 1;
      else
        bsizes[ilen] += 1;
      for (std::size_t j = i + 1; j < order; ++j)
        EXPECT_TRUE(are_disjoint(blocks[i], blocks[j]));
    }
    // check that min block offset is zero (check on total coverage of axis is
    // implicit in the results of this test, the previous test of disjointness,
    // and following tests)
    EXPECT_EQ(minb, 0);
    // check that distribution of block sizes is correct
    ASSERT_TRUE(bsizes.count(axis_length / order) > 0);
    std::size_t n0 = bsizes[axis_length / order];
    bsizes.erase(axis_length / order);
    EXPECT_EQ(n0, order - axis_length % order);
    std::size_t n1 = 0;
    if (bsizes.count(axis_length / order + 1) > 0) {
      n1 = bsizes[axis_length / order + 1];
      bsizes.erase(axis_length / order + 1);
    }
    EXPECT_EQ(n1, axis_length % order);
    EXPECT_EQ(bsizes.size(), 0);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
