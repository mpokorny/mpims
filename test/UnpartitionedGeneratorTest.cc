#include <optional>
#include <stdexcept>
#include <tuple>

#include "BlockGenerator.h"
#include "gtest/gtest.h"

using namespace mpims;

TEST(UnpartitionedGenerator, TotalSequence) {

  const std::size_t axis_length = 25;
  auto st = UnpartitionedGenerator::initial_states(axis_length)(0);
  std::optional<block_t> blk;
  std::tie(st, blk) = UnpartitionedGenerator::apply(st);
  ASSERT_TRUE(blk);
  std::size_t b0, blen;
  std::tie(b0, blen) = blk.value();
  EXPECT_EQ(b0, 0);
  EXPECT_EQ(blen, axis_length);
  std::tie(st, blk) = UnpartitionedGenerator::apply(st);
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
