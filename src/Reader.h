/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef READER_H_
#define READER_H_

#include <mpi.h>

#include <complex>
#include <deque>
#include <iostream>
#include <iterator>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <ColumnAxis.h>
#include <IndexBlock.h>
#include <MSArray.h>
#include <MSColumns.h>
#include <DataDistribution.h>
#include <MPIState.h>
#include <TraversalState.h>
#include <ReaderBase.h>

namespace mpims {

enum AMode {
  WriteOnly,
  ReadWrite
};

class IterParams;
class AxisIter;

class Reader
  : public ReaderBase
  , public std::iterator<std::input_iterator_tag, const MSArray, std::size_t> {

  friend class Writer;

public:

  Reader()
    : m_readahead(false) {
  }

  Reader(
    MPIState&& mpi_state,
    const std::string& datarep,
    std::shared_ptr<const std::vector<ColumnAxisBase<MSColumns> > > ms_shape,
    std::shared_ptr<const std::vector<IterParams> > iter_params,
    std::shared_ptr<const std::vector<MSColumns> > buffer_order,
    std::shared_ptr<const std::optional<MSColumns> > inner_fileview_axis,
    std::shared_ptr<const ArrayIndexer<MSColumns> > ms_indexer,
    std::size_t buffer_size,
    std::size_t value_extent,
    bool readahead,
    TraversalState&& traversal_state,
    bool debug_log);

  Reader(const Reader& other)
    : m_mpi_state(other.m_mpi_state)
    , m_datarep(other.m_datarep)
    , m_ms_shape(other.m_ms_shape)
    , m_iter_params(other.m_iter_params)
    , m_buffer_order(other.m_buffer_order)
    , m_inner_fileview_axis(other.m_inner_fileview_axis)
    , m_etype_datatype(other.m_etype_datatype)
    , m_ms_indexer(other.m_ms_indexer)
    , m_rank(other.m_rank)
    , m_buffer_size(other.m_buffer_size)
    , m_value_extent(other.m_value_extent)
    , m_readahead(other.m_readahead)
    , m_debug_log(other.m_debug_log)
    , m_traversal_state(other.m_traversal_state)
    , m_next_traversal_state(other.m_next_traversal_state)
    , m_ms_array(other.m_ms_array)
    , m_next_ms_array(other.m_next_ms_array) {
  }

  Reader(Reader&& other)
    : m_mpi_state(std::move(other).m_mpi_state)
    , m_datarep(std::move(other).m_datarep)
    , m_ms_shape(std::move(other).m_ms_shape)
    , m_iter_params(std::move(other).m_iter_params)
    , m_buffer_order(std::move(other).m_buffer_order)
    , m_inner_fileview_axis(std::move(other).m_inner_fileview_axis)
    , m_etype_datatype(std::move(other).m_etype_datatype)
    , m_ms_indexer(std::move(other).m_ms_indexer)
    , m_rank(std::move(other).m_rank)
    , m_buffer_size(std::move(other).m_buffer_size)
    , m_value_extent(std::move(other).m_value_extent)
    , m_readahead(std::move(other).m_readahead)
    , m_debug_log(std::move(other).m_debug_log)
    , m_traversal_state(std::move(other).m_traversal_state)
    , m_next_traversal_state(std::move(other).m_next_traversal_state)
    , m_ms_array(std::move(other).m_ms_array)
    , m_next_ms_array(std::move(other).m_next_ms_array) {
  }

  Reader&
  operator=(const Reader& other) {
    if (this != &other) {
      Reader temp(other);
      std::lock_guard<decltype(m_mtx)> lock(m_mtx);
      swap(temp);
    }
    return *this;
  }

  Reader&
  operator=(Reader&& other) {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    m_mpi_state = std::move(other).m_mpi_state;
    m_datarep = std::move(other).m_datarep;
    m_ms_shape = std::move(other).m_ms_shape;
    m_rank = std::move(other).m_rank;
    m_buffer_size = std::move(other).m_buffer_size;
    m_value_extent = std::move(other).m_value_extent;
    m_readahead = std::move(other).m_readahead;
    m_iter_params = std::move(other).m_iter_params;
    m_buffer_order = std::move(other).m_buffer_order;
    m_inner_fileview_axis = std::move(other).m_inner_fileview_axis;
    m_etype_datatype = std::move(other).m_etype_datatype;
    m_ms_indexer = std::move(other).m_ms_indexer;
    m_debug_log = std::move(other).m_debug_log;
    m_traversal_state = std::move(other).m_traversal_state;
    m_next_traversal_state = std::move(other).m_next_traversal_state;
    m_ms_array = std::move(other).m_ms_array;
    m_next_ms_array = std::move(other).m_next_ms_array;
    return *this;
  }

  ~Reader();

  static Reader
  begin(
    const std::string& path,
    const std::string& datarep,
    MPI_Comm comm,
    MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    bool ms_buffer_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool readahead,
    bool debug_log = false);

  static const Reader
  end() {
    return Reader();
  };

  bool
  operator==(const Reader& other) const {
    if (other.at_end())
      return at_end();
    // we don't assume here that file names and contents have remained
    // unmodified since creating either Reader
    return (
      m_ms_shape == other.m_ms_shape
      && m_inner_fileview_axis == other.m_inner_fileview_axis
      && m_iter_params == other.m_iter_params
      && m_buffer_order == other.m_buffer_order
      && m_traversal_state == other.m_traversal_state
      // and now the potentially really expensive test -- this is required by
      // semantics of an InputIterator
      && buffer_length() == other.buffer_length()
      && getv() == other.getv());
  }

  bool
  operator!=(const Reader& other) const {
    return !operator==(other);
  }

  unsigned
  num_ranks() const {
    unsigned result;
    auto handles = m_mpi_state.handles();
    std::lock_guard<MPIHandles> lck(*handles);
    if (handles->comm != MPI_COMM_NULL)
      MPI_Comm_size(handles->comm, reinterpret_cast<int*>(&result));
    else
      result = 0;
    return result;
  }

  bool
  at_end() const;

  void
  next() {
    step(true);
  }

  void
  interrupt() {
    step(false);
  }

  const std::vector<IndexBlockSequence<MSColumns> >&
  indices() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    m_ms_array.wait();
    return m_ms_array.blocks();
  };

  std::size_t
  buffer_length() const;

  void
  swap(Reader& other);

  Reader&
  operator++() {
    next();
    return *this;
  }

  Reader
  operator++(int) {
    Reader result(*this);
    operator++();
    return result;
  }

  // NB: the returned array can be empty!
  const MSArray&
  operator*() const {
    return getv();
  }

  const MSArray*
  operator->() const {
    return &getv();
  }

protected:

  struct SetFileviewArgs {
    ArrayIndexer<MSColumns>::index data_index;
    bool in_tail;
    MPI_Datatype datatype;
    MPI_File file;
  };

  static Reader
  wbegin(
    const std::string& path,
    const std::string& datarep,
    AMode access_mode,
    MPI_Comm comm,
    MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    bool ms_buffer_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool debug_log = false);

  static void
  initialize(
    const std::string& path,
    const std::string& datarep,
    MPI_Comm comm,
    MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    bool ms_buffer_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool debug_log,
    int amode,
    MPI_Comm& reduced_comm,
    MPI_Info& priv_info,
    MPI_File& file,
    std::size_t& value_extent,
    std::shared_ptr<std::vector<IterParams> >& iter_params,
    std::shared_ptr<std::vector<MSColumns> >& buffer_order,
    std::shared_ptr<std::optional<MSColumns> >& inner_fileview_axis,
    std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
    std::size_t& buffer_size,
    TraversalState& traversal_state);

  void
  step(bool cont);

  void
  start_next();

  const MSArray&
  getv() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    m_ms_array.wait();
    return m_ms_array;
  };

  static std::tuple<std::unique_ptr<MPI_Datatype, DatatypeDeleter>, unsigned>
  init_buffer_datatype(
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::shared_ptr<std::vector<IterParams> >& iter_params,
    bool ms_buffer_order,
    bool tail_array,
    int rank,
    bool debug_log);

  static std::unique_ptr<MPI_Datatype, DatatypeDeleter>
  init_fileview(
    MPI_Aint value_extent,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::shared_ptr<std::vector<IterParams> >& iter_params,
    const std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
    bool tail_fileview,
    int rank,
    bool debug_log);

  MSArray
  read_arrays(
    TraversalState& traversal_state,
    bool nonblocking,
    std::vector<IndexBlockSequence<MSColumns> >& blocks,
    MPI_File file) const;

  void
  set_fileview(const SetFileviewArgs& args) const;

  void
  set_deferred_fileview() const;

  void
  advance_to_next_buffer(
    TraversalState& traversal_state,
    MPI_File file) const;

  void
  extend();

  MSArray
  read_buffer(
    TraversalState& TraversalState,
    bool nonblocking,
    MPI_File file) const;

  mutable std::shared_ptr<const MPI_Datatype> m_buffer_datatype;

  bool
  buffer_order_compare(const MSColumns& col0, const MSColumns& col1) const;

  std::vector<IndexBlockSequence<MSColumns> >
  sorted_blocks(const TraversalState& state) const;

private:

  MPIState m_mpi_state;

  std::string m_datarep;

  std::shared_ptr<const std::vector<ColumnAxisBase<MSColumns> > > m_ms_shape;

  // ms array iteration definition, for this rank, in traversal order
  std::shared_ptr<const std::vector<IterParams> > m_iter_params;

  std::shared_ptr<const std::vector<MSColumns> > m_buffer_order;

  std::shared_ptr<const std::optional<MSColumns> > m_inner_fileview_axis;

  std::shared_ptr<const MPI_Datatype> m_etype_datatype;

  std::shared_ptr<const ArrayIndexer<MSColumns> > m_ms_indexer;

  int m_rank;

  std::size_t m_buffer_size;

  std::size_t m_value_extent;

  bool m_readahead;

  bool m_debug_log;

  TraversalState m_traversal_state;

  TraversalState m_next_traversal_state;

  mutable MSArray m_ms_array;

  mutable MSArray m_next_ms_array;

  mutable std::recursive_mutex m_mtx;

  mutable std::optional<SetFileviewArgs> m_deferred_fileview_args;
};

void
swap(Reader& r1, Reader& r2);

} // end namespace mpims

#endif // READER_H_
