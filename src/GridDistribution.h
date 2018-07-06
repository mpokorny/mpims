#ifndef GRID_DISTRIBUTION_H_
#define GRID_DISTRIBUTION_H_

#include <functional>
#include <memory>
#include <optional>

#include <DataDistribution.h>

namespace mpims {

typedef std::function<
  std::shared_ptr<const DataDistribution>(const std::optional<std::size_t>&)>
  GridDistribution;

class GridDistributionFactory {

public:

  static GridDistribution
  cyclic(std::size_t block_length, std::size_t order) {

    return [=](const std::optional<std::size_t>& len) {
      return DataDistributionFactory::cyclic(block_length, order, len);
    };
  }

  static GridDistribution
  block_sequence(const std::vector<std::vector<finite_block_t> >& all_blocks) {

    return [=](const std::optional<std::size_t>& len) {
      return DataDistributionFactory::block_sequence(all_blocks, len);
    };
  }

  static GridDistribution
  unpartitioned() {

    return [=](const std::optional<std::size_t>& len) {
      return DataDistributionFactory::unpartitioned(len);
    };
  }
};

} // end namespace mpims

#endif // GRID_DISTRIBUTION_H_
// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
