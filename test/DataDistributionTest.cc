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

TEST(DataDistribution, AllBlocks) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  for (std::size_t gi = 0; gi < group_size; ++gi) {
    std::vector<block_t> blks;
    for (std::size_t i = gi * block_size;
         i < axis_length;
         i += block_size * group_size)
      blks.push_back(block_t(i, std::min(i + block_size, axis_length) - i));
    auto dd =
      DataDistributionFactory::cyclic(block_size, group_size, axis_length, gi);
    EXPECT_EQ(dd->blocks(), blks);
  }
}

TEST(DataDistribution, IteratorLifecycle) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  std::unique_ptr<DataDistribution::Iterator> it;
  {
    auto dd =
      DataDistributionFactory::cyclic(block_size, group_size, axis_length, 0);
    it = dd->begin();
  }
  EXPECT_NO_THROW(**it);
  EXPECT_NO_THROW(++(*it));
  EXPECT_NO_THROW(**it);
}

TEST(DataDistribution, IteratePastEnd) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  auto it =
    DataDistributionFactory::cyclic(block_size, group_size, axis_length, 0)->
    begin();
  EXPECT_NO_THROW(**it);
  EXPECT_NO_THROW(it->take_all());
  EXPECT_ANY_THROW(**it);
}

TEST(DataDistribution, Iteration) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;

  for (std::size_t gi = 0; gi < group_size; ++gi) {
    std::vector<size_t> indices;
    for (std::size_t i = gi * block_size;
         i < axis_length;
         i += block_size * group_size) {
      std::size_t len = std::min(i + block_size, axis_length) - i;
      for (std::size_t j = 0; j < len; ++j)
        indices.push_back(i + j);
    }
    auto it =
      DataDistributionFactory::cyclic(block_size, group_size, axis_length, gi)->
      begin();
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

  for (std::size_t gi = 0; gi < group_size; ++gi) {
    std::vector<size_t> indices;
    for (std::size_t i = gi * block_size;
         i < axis_length;
         i += block_size * group_size) {
      std::size_t len = std::min(i + block_size, axis_length) - i;
      for (std::size_t j = 0; j < len; ++j)
        indices.push_back(i + j);
    }
    auto dd =
      DataDistributionFactory::cyclic(block_size, group_size, axis_length, gi);
    auto it = dd->begin();
    EXPECT_TRUE(
      std::all_of(
        std::begin(indices),
        std::end(indices),
        [&it](auto& i) {
          return it->take() == std::vector<std::size_t>{i};
        }));
    EXPECT_TRUE(it->at_end());

    it = dd->begin();
    EXPECT_EQ(it->take_all(), indices);
    EXPECT_TRUE(it->at_end());
  }
}

TEST(DataDistribution, IteratorTakeBlocked) {

  const std::size_t block_size = 3, group_size = 2, axis_length = 14;
  const std::size_t group_len = block_size * group_size;

  for (std::size_t gi = 0; gi < group_size; ++gi) {
    std::vector<size_t> indices;
    for (std::size_t i = gi * block_size;
         i < axis_length;
         i += block_size * group_size) {
      std::size_t len = std::min(i + block_size, axis_length) - i;
      for (std::size_t j = 0; j < len; ++j)
        indices.push_back(i + j);
    }

    auto dd =
      DataDistributionFactory::cyclic(block_size, group_size, axis_length, gi);

    for (
      std::size_t tksz = 1;
      tksz < ceil(axis_length, group_len) * block_size;
      ++tksz) {

      auto it = dd->begin();

      auto idx = std::begin(indices);
      while (idx != std::end(indices)) {
        std::size_t n = 0;
        for (auto& blk : it->take_blocked(tksz)) {
          std::size_t b0, blen;
          std::tie(b0, blen) = blk;
          for (std::size_t i = 0; idx != std::end(indices) && i < blen; ++i) {
            EXPECT_EQ(b0 + i, *idx);
            ++idx;
          }
          n += blen;
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
