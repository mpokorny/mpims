/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
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
      : std::runtime_error("indeterminate internal array axis") {
    }
  };

  // Traversal of an indeterminate MS with the top traversal axis not equal to
  // the top (indeterminate) MS axis.
  class IndeterminateArrayTraversalError
    : std::runtime_error {
  public:
    explicit IndeterminateArrayTraversalError()
      : std::runtime_error(
        "top of traversal order not the indeterminate axis") {
    }
  };

  static void
  init_iterparams(
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    const std::unordered_map<MSColumns, DataDistribution>& pgrid,
    int rank,
    bool debug_log,
    std::shared_ptr<std::vector<IterParams> >& iter_params);

  static void
  init_traversal_partitions(
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
  vector_datatype(
    MPI_Aint value_extent,
    std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
    std::size_t dt_extent,
    std::size_t offset,
    std::size_t num_blocks,
    std::size_t block_len,
    std::size_t terminal_block_len,
    std::size_t stride,
    std::size_t len,
    int rank,
    bool debug_log);

  static std::tuple<
    std::unique_ptr<MPI_Datatype, DatatypeDeleter>,
    std::size_t,
    bool>
  compound_datatype(
    MPI_Aint value_extent,
    std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
    std::size_t dt_extent,
    std::size_t offset,
    std::size_t stride,
    std::size_t num_blocks,
    std::size_t block_len,
    std::size_t terminal_block_len,
    const std::optional<std::size_t>& len,
    int rank,
    bool debug_log);

  static std::tuple<std::optional<std::tuple<std::size_t, std::size_t> >, bool>
  tail_buffer_blocks(const IterParams& ip);

  static std::unique_ptr<std::vector<IndexBlockSequenceMap<MSColumns> > >
  make_index_block_sequences(
    const std::shared_ptr<const std::vector<IterParams> >& iter_params);

  static const IterParams*
  find_iter_params(
    const std::shared_ptr<const std::vector<IterParams> >& iter_params,
    MSColumns col);
};

} // end namespace mpims

#endif // READER_BASE_H_
