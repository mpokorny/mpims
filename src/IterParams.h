#ifndef ITER_PARAMS_H_
#define ITER_PARAMS_H_

#include <memory>
#include <optional>

#include <mpims.h>
#include <MSColumns.h>

#include <mpi.h>

#include "DataDistribution.h"

namespace mpims {

// IterParams holds the information for the iteration over a single axis, as
// well as context within the scope of iteration over the entire MS together
// given a user buffer size. Most of the values in this structure are used to
// describe the iteration over an axis for a particular process.
struct IterParams {

  IterParams() {
  }

  IterParams(
    MSColumns axis_,
    const std::optional<std::size_t>& axis_length_,
    const std::shared_ptr<const DataDistribution>& data_distribution_,
    std::size_t rank_)
    : axis(axis_)
    , axis_length(axis_length_)
    , data_distribution(data_distribution_)
    , rank(rank_) {

    full_fv_axis = false;
    fully_in_array = false;
    within_fileview = false;
    buffer_capacity = 0;
    array_length = 1;
    origin = **begin();
  }

  MSColumns axis;
  std::optional<std::size_t> axis_length;
  std::size_t full_fv_axis;
  // can a buffer hold the data for the full axis given iteration pattern?
  bool fully_in_array;
  // is iteration across this axis done without changing the fileview?
  bool within_fileview;
  // number of axis values in iteration for which the entire array comprising
  // data from all deeper axes can be copied into a single buffer
  std::size_t buffer_capacity;
  // number of data values in array comprising data from all deeper axes
  std::size_t array_length;
  // remainder are values describing iteration pattern
  std::shared_ptr<const DataDistribution> data_distribution;
  std::size_t rank;
  std::size_t origin;

  std::optional<std::size_t>
  size() const {
    return data_distribution->size(rank);
  }

  std::optional<std::size_t>
  max_size() const {
    return data_distribution->max_size();
  }

  std::unique_ptr<DataDistribution::Iterator>
  begin() const {
    return data_distribution->begin(rank);
  }

  std::optional<std::size_t>
  period() const {
    return data_distribution->period();
  }

  std::size_t
  period_max_factor_size() const {
    return
      map(
        period(),
        [this](auto p) {
          return begin()->take_while([&p](auto& i) { return i < p; }).size(); 
        }).value_or(1);
  }

  bool
  selection_repeats_uniformly(MPI_Comm comm) {

    bool my_result;
    auto sz = size();
    if (buffer_capacity == 0 || (sz && buffer_capacity >= sz.value())) {
      my_result = true;
    } else {
      auto p = period();
      if (!p || (axis_length && axis_length.value() % p.value() != 0)) {
        my_result = false;
      } else {
        std::size_t elements_per_period =
          begin()->take_while([&p](auto& i) { return i < p; }).size();
        my_result = buffer_capacity % elements_per_period == 0;
      }
    }
    bool result;
    if (comm != MPI_COMM_NULL)
      MPI_Allreduce(&my_result, &result, 1, MPI_CXX_BOOL, MPI_LAND, comm);
    else
      result = my_result;
    return result;
  }

  bool
  operator==(const IterParams& rhs) const {
    return (
      axis == rhs.axis
      && axis_length == rhs.axis_length
      && fully_in_array == rhs.fully_in_array
      && within_fileview == rhs.within_fileview
      && buffer_capacity == rhs.buffer_capacity
      && rank == rhs.rank
      && *data_distribution == *rhs.data_distribution);
  }

  bool
  operator!=(const IterParams& rhs) const {
    return !(operator==(rhs));
  }
};

}

#endif // #define ITER_PARAMS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
