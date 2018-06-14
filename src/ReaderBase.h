#ifndef READER_BASE_H_
#define READER_BASE_H_

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include <mpims.h>
#include <ColumnAxis.h>
#include <DataDistribution.h>
#include <IndexBlock.h>
#include <IterParams.h>
#include <MSColumns.h>

namespace mpims {

struct ReaderBase {

  // Array with an indeterminate, internal, that is, any but the outermost, axis
  class IndeterminateArrayError
    : std::runtime_error {
  public:
    explicit IndeterminateArrayError()
      : std::runtime_error("indeterminate internal array axis") {}
  };

  // Traversal of an indeterminate MS with the top traversal axis not equal to
  // the top (indeterminate) MS axis.
  class IndeterminateArrayTraversalError
    : std::runtime_error {
  public:
    explicit IndeterminateArrayTraversalError()
      : std::runtime_error(
        "top of traversal order not the indeterminate axis") {}
  };

  // Complex axis length cannot be anything but two.
  class ComplexAxisLengthError
    : std::runtime_error {
  public:
    explicit ComplexAxisLengthError()
      : std::runtime_error("complex axis length is not two") {}
  };

  // Specification of the "complex" axis is only allowed when the Reader/Writer
  // template type is not complex-valued.
  class ComplexAxisError
    : std::runtime_error {
  public:
    explicit ComplexAxisError()
      : std::runtime_error(
        "complex axis cannot be used with complex-valued data type") {}
  };

  static void
  init_iterparams(
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    const std::unordered_map<
      MSColumns,
      std::shared_ptr<const DataDistribution> >& pgrid,
    int rank,
    std::shared_ptr<std::vector<IterParams> >& iter_params);

  static void
  init_traversal_partitions(
    MPI_Comm comm,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    std::size_t& buffer_size,
    std::shared_ptr<std::vector<IterParams> >& iter_params,
    std::shared_ptr<std::optional<MSColumns> >& inner_fileview_axis,
    std::size_t value_size,
    int rank,
    bool debug_log);

  static std::tuple<
    std::unique_ptr<MPI_Datatype, DatatypeDeleter>,
    std::size_t>
  finite_compound_datatype(
    std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
    std::size_t dt_extent,
    const std::vector<finite_block_t>& blocks,
    std::size_t len,
    int rank,
    bool debug_log);

  static const IterParams*
  find_iter_params(
    const std::shared_ptr<const std::vector<IterParams> >& iter_params,
    MSColumns col);
};

} // end namespace mpims

#endif // READER_BASE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
