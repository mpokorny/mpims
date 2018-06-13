#include <algorithm>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "DataDistribution.h"
#include "gtest/gtest.h"

using namespace mpims;

template <
  typename T,
  typename = typename std::enable_if<std::is_unsigned<T>::value>::type >
T
ceil(T num, T denom) {
  return (num + (denom - 1)) / denom;
}

TEST(DataDistribution, OrderValue) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;
  auto cy =
    DataDistributionFactory::cyclic(block_size, group_size, axis_length);
  EXPECT_EQ(cy->order(), group_size);

  const std::vector<std::vector<block_t> > all_blocks{
    std::vector{block_t(0, 2), block_t(5, 3), block_t(12, 1)},
      std::vector{block_t(2, 2), block_t(8, 3), block_t(13, 1)}};
  auto bs =
    DataDistributionFactory::block_sequence(all_blocks, axis_length);
  EXPECT_EQ(bs->order(), all_blocks.size());
}

TEST(DataDistribution, AllBlocks) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  std::size_t max_size = 0;
  auto dd =
    DataDistributionFactory::cyclic(block_size, group_size, axis_length);
  for (std::size_t rank = 0; rank < dd->order(); ++rank) {
    std::vector<block_t> blks;
    std::size_t size = 0;
    for (std::size_t i = rank * block_size;
         i < axis_length;
         i += block_size * group_size) {
      auto sz = std::min(i + block_size, axis_length) - i;
      blks.push_back(block_t(i, sz));
      size += sz;
    }
    EXPECT_EQ(dd->blocks(rank), blks);
    EXPECT_EQ(dd->size(rank), size);
    max_size = std::max(max_size, size);
  }
  EXPECT_EQ(dd->max_size(), max_size);

  // can't get all blocks of unbounded distribution
  dd = DataDistributionFactory::unpartitioned();
  EXPECT_THROW(dd->blocks(0), std::domain_error);
  EXPECT_FALSE(dd->size(0));
}

TEST(DataDistribution, IteratorLifecycle) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  std::unique_ptr<DataDistribution::Iterator> it0, it1;
  {
    auto dd =
      DataDistributionFactory::cyclic(block_size, group_size, axis_length);
    it0 = dd->begin(0);
    it1 = dd->begin(1);
  }
  EXPECT_NO_THROW(**it0);
  EXPECT_NO_THROW(++(*it0));
  EXPECT_NO_THROW(**it0);
  it0.reset();

  EXPECT_NO_THROW(**it1);
  EXPECT_NO_THROW(++(*it1));
  EXPECT_NO_THROW(**it1);
}

TEST(DataDistribution, IteratePastEnd) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  auto it =
    DataDistributionFactory::cyclic(block_size, group_size, axis_length)->
    begin(0);
  EXPECT_NO_THROW(**it);
  EXPECT_NO_THROW(it->take_all());
  EXPECT_ANY_THROW(**it);
}

TEST(DataDistribution, Iteration) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  auto dd =
    DataDistributionFactory::cyclic(block_size, group_size, axis_length);
  for (std::size_t rank = 0; rank < dd->order(); ++rank) {
    std::vector<size_t> indices;
    for (std::size_t i = rank * block_size;
         i < axis_length;
         i += block_size * group_size) {
      std::size_t len = std::min(i + block_size, axis_length) - i;
      for (std::size_t j = 0; j < len; ++j)
        indices.push_back(i + j);
    }
    auto it = dd->begin(rank);
    EXPECT_TRUE(
      std::all_of(
        std::begin(indices),
        std::end(indices),
        [&it](auto& i) {
          bool result = **it == i;
          ++(*it);
          return result;
        }));
    EXPECT_TRUE(it->at_end());
  }
}

TEST(DataDistribution, IteratorTake) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  auto dd =
    DataDistributionFactory::cyclic(block_size, group_size, axis_length);
  for (std::size_t rank = 0; rank < dd->order(); ++rank) {
    std::vector<size_t> indices;
    for (std::size_t i = rank * block_size;
         i < axis_length;
         i += block_size * group_size) {
      std::size_t len = std::min(i + block_size, axis_length) - i;
      for (std::size_t j = 0; j < len; ++j)
        indices.push_back(i + j);
    }
    auto it = dd->begin(rank);
    EXPECT_TRUE(
      std::all_of(
        std::begin(indices),
        std::end(indices),
        [&it](auto& i) {
          return it->take() == std::vector<std::size_t>{i};
        }));
    EXPECT_TRUE(it->at_end());

    it = dd->begin(rank);
    EXPECT_EQ(it->take_all(), indices);
    EXPECT_TRUE(it->at_end());
  }
}

TEST(DataDistribution, IteratorTakeBlocked) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;
  const std::size_t group_len = block_size * group_size;

  auto dd =
    DataDistributionFactory::cyclic(block_size, group_size, axis_length);

  for (std::size_t rank = 0; rank < dd->order(); ++rank) {
    std::vector<size_t> indices;
    for (std::size_t i = rank * block_size;
         i < axis_length;
         i += block_size * group_size) {
      std::size_t len = std::min(i + block_size, axis_length) - i;
      for (std::size_t j = 0; j < len; ++j)
        indices.push_back(i + j);
    }

    for (
      std::size_t tksz = 1;
      tksz < ceil(axis_length, group_len) * block_size;
      ++tksz) {

      auto it = dd->begin(rank);

      auto idx = std::begin(indices);
      while (idx != std::end(indices)) {
        std::size_t n = 0;
        for (auto& blk : it->take_blocked(tksz)) {
          std::size_t b0;
          std::optional<std::size_t> blen;
          std::tie(b0, blen) = blk;
          for (
            std::size_t i = 0;
            idx != std::end(indices) && i < blen.value();
            ++i) {
            EXPECT_EQ(b0 + i, *idx);
            ++idx;
          }
          n += blen.value();
        }
        if (!it->at_end()) {
          EXPECT_EQ(n, tksz);
        } else {
          EXPECT_EQ(n, tksz - (tksz - indices.size() % tksz) % tksz);
        }
      }
      EXPECT_TRUE(it->at_end());
    }
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
