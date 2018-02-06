/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef READER_H_
#define READER_H_

#include <mpi.h>

#include <complex>
#include <iostream>
#include <iterator>
#include <mutex>
#include <optional>
#include <stack>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <ColumnAxis.h>
#include <IndexBlock.h>
#include <MSArray.h>
#include <MSColumns.h>
#include <DataDistribution.h>
#include <ReaderMPIState.h>

namespace mpims {

class Reader
  : public std::iterator<std::input_iterator_tag, const MSArray, std::size_t> {

public:

  class IterParams;
  class AxisIter;


  struct TraversalState {

    TraversalState()
      : eof(true)
      , cont(false) {
    }

    TraversalState(
      ::MPI_Comm comm,
      const std::shared_ptr<const std::vector<IterParams> >& iter_params_,
      MSColumns& outer_full_array_axis_,
      const std::shared_ptr<const ::MPI_Datatype>& full_array_datatype,
      unsigned full_array_dt_count,
      const std::shared_ptr<const ::MPI_Datatype>& tail_array_datatype,
      unsigned tail_array_dt_count,
      const std::shared_ptr<const ::MPI_Datatype>& full_fileview_datatype,
      const std::shared_ptr<const ::MPI_Datatype>& tail_fileview_datatype)
      : eof(comm == MPI_COMM_NULL)
      , cont(!eof)
      , iter_params(iter_params_)
      , block_maps(make_index_block_sequences(iter_params_))
      , count(0)
      , max_count(0)
      , in_tail(false)
      , outer_full_array_axis(outer_full_array_axis_)
      , m_full_array_datatype(full_array_datatype)
      , m_full_array_dt_count(full_array_dt_count)
      , m_tail_array_datatype(tail_array_datatype)
      , m_tail_array_dt_count(tail_array_dt_count)
      , m_full_fileview_datatype(full_fileview_datatype)
      , m_tail_fileview_datatype(tail_fileview_datatype) {
    }

    bool eof;

    bool cont;

    std::shared_ptr<const std::vector<IterParams> > iter_params;

    std::shared_ptr<
      const std::vector<IndexBlockSequenceMap<MSColumns> > > block_maps;

    ArrayIndexer<MSColumns>::index data_index;

    std::stack<AxisIter> axis_iters;

    int count;

    int max_count;

    bool in_tail;

    MSColumns outer_full_array_axis;

    std::tuple<std::shared_ptr<const ::MPI_Datatype>, unsigned>
    array_datatype() const {
      if (in_tail)
        return std::make_tuple(m_tail_array_datatype, m_tail_array_dt_count);
      else
        return std::make_tuple(m_full_array_datatype, m_full_array_dt_count);
    }

    std::shared_ptr<const ::MPI_Datatype>
    fileview_datatype() const {
      return (in_tail ? m_tail_fileview_datatype : m_full_fileview_datatype);
    }

    bool
    operator==(const TraversalState& other) const {
      return (
        eof == other.eof
        && cont == other.cont
        && count == other.count
        && (block_maps == other.block_maps || *block_maps == *other.block_maps)
        && data_index == other.data_index
        && axis_iters == other.axis_iters
        && in_tail == other.in_tail
        && m_tail_array_dt_count == other.m_tail_array_dt_count
        && m_full_array_dt_count == other.m_full_array_dt_count
        // comparing array datatypes and fileview datatypes would be ideal, but
        // that could be a relatively expensive task, as it requires digging
        // into the contents of the datatypes; instead, since those datatypes
        // are functions of the MS shape, iteration order and outer array axis
        // alone, given the previous comparisons, we can just compare the outer
        // arrays axis values
        && outer_full_array_axis == other.outer_full_array_axis);
    }

    bool
    operator!=(const TraversalState& other) const {
      return !operator==(other);
    }

    std::vector<IndexBlockSequence<MSColumns> >
    blocks() const {
      std::vector<IndexBlockSequence<MSColumns> > result;
      if (count > 0) {
        for (std::size_t i = 0; i < iter_params->size(); ++i) {
          auto& ip = (*iter_params)[i];
          if (!ip.fully_in_array && ip.buffer_capacity == 0) {
            result.emplace_back(
              ip.axis,
              std::vector{ IndexBlock(data_index.at(ip.axis), 1) });
          } else {
            auto& map = (*block_maps)[i];
            auto ibs = map[data_index.at(ip.axis)];
            ibs.trim();
            result.push_back(ibs);
          }
        }
      }
      return result;
    }

  private:

    std::shared_ptr<const ::MPI_Datatype> m_full_array_datatype;

    unsigned m_full_array_dt_count;

    std::shared_ptr<const ::MPI_Datatype> m_tail_array_datatype;

    unsigned m_tail_array_dt_count;

    std::shared_ptr<const ::MPI_Datatype> m_full_fileview_datatype;

    std::shared_ptr<const ::MPI_Datatype> m_tail_fileview_datatype;

  };

  Reader();

  Reader(
    ReaderMPIState&& mpi_state,
    std::shared_ptr<const std::vector<ColumnAxisBase<MSColumns> > > ms_shape,
    std::shared_ptr<const std::vector<IterParams> > iter_params,
    std::shared_ptr<const std::vector<MSColumns> > buffer_order,
    std::shared_ptr<const std::optional<MSColumns> > inner_fileview_axis,
    std::shared_ptr<const ArrayIndexer<MSColumns> > ms_indexer,
    std::size_t buffer_size,
    bool readahead,
    TraversalState&& traversal_state,
    bool debug_log);

  Reader(const Reader& other)
    : m_mpi_state(other.m_mpi_state)
    , m_ms_shape(other.m_ms_shape)
    , m_iter_params(other.m_iter_params)
    , m_buffer_order(other.m_buffer_order)
    , m_inner_fileview_axis(other.m_inner_fileview_axis)
    , m_etype_datatype(other.m_etype_datatype)
    , m_ms_indexer(other.m_ms_indexer)
    , m_rank(other.m_rank)
    , m_buffer_size(other.m_buffer_size)
    , m_readahead(other.m_readahead)
    , m_debug_log(other.m_debug_log)
    , m_traversal_state(other.m_traversal_state)
    , m_next_traversal_state(other.m_next_traversal_state)
    , m_ms_array(other.m_ms_array)
    , m_next_ms_array(other.m_next_ms_array) {
  }

  Reader(Reader&& other)
    : m_mpi_state(std::move(other).m_mpi_state)
    , m_ms_shape(std::move(other).m_ms_shape)
    , m_iter_params(std::move(other).m_iter_params)
    , m_buffer_order(std::move(other).m_buffer_order)
    , m_inner_fileview_axis(std::move(other).m_inner_fileview_axis)
    , m_etype_datatype(std::move(other).m_etype_datatype)
    , m_ms_indexer(std::move(other).m_ms_indexer)
    , m_rank(std::move(other).m_rank)
    , m_buffer_size(std::move(other).m_buffer_size)
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
      std::lock_guard<decltype(m_mtx)> lock(m_mtx);
      Reader temp(other);
      swap(temp);
    }
    return *this;
  }

  Reader&
  operator=(Reader&& other) {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    m_mpi_state = std::move(other).m_mpi_state;
    m_ms_shape = std::move(other).m_ms_shape;
    m_rank = std::move(other).m_rank;
    m_buffer_size = std::move(other).m_buffer_size;
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

  struct IterParams {
    MSColumns axis;
    bool fully_in_array, within_fileview;
    std::size_t buffer_capacity, array_length;
    std::size_t length, origin, stride, block_len, max_blocks,
      terminal_block_len, max_terminal_block_len;

    bool
    operator==(const IterParams& rhs) const {
      return (
        axis == rhs.axis
        && fully_in_array == rhs.fully_in_array
        && within_fileview == rhs.within_fileview
        && buffer_capacity == rhs.buffer_capacity
        && length == rhs.length
        && origin == rhs.origin
        && stride == rhs.stride
        && block_len == rhs.block_len
        && max_blocks == rhs.max_blocks
        && terminal_block_len == rhs.terminal_block_len
        && max_terminal_block_len == rhs.max_terminal_block_len);
    }

    bool
    operator!=(const IterParams& rhs) const {
      return !(operator==(rhs));
    }

    std::size_t
    true_length() const {
      return block_len * (max_blocks - 1) + terminal_block_len;
    }
  };

  class AxisIter {
  public:

    AxisIter(std::shared_ptr<const IterParams> params_, bool outer_at_data_)
      : params(params_)
      , index(params_->origin)
      , block(0)
      , at_data(params_->max_blocks > 0 && outer_at_data_)
      , outer_at_data(outer_at_data_)
      , at_end(params_->max_blocks == 0) {
    }

    bool
    operator==(const AxisIter& rhs) const {
      return *params == *rhs.params && index == rhs.index && block == rhs.block
        && at_data == rhs.at_data && outer_at_data == rhs.outer_at_data
        && at_end == rhs.at_end;
    }

    bool
    operator!=(const AxisIter& rhs) const {
      return !(operator==(rhs));
    }

    std::shared_ptr<const IterParams> params;
    std::size_t index;
    std::size_t block;
    bool at_data;
    bool outer_at_data;
    bool at_end;

    void
    increment(std::size_t n=1) {
      while (n > 0 && !at_end) {
        block = index / params->stride;
        auto block_origin = params->origin + block * params->stride;
        bool terminal_block = block == params->max_blocks - 1;
        auto block_len =
          (terminal_block ? params->terminal_block_len : params->block_len);
        ++index;
        if (!terminal_block && index - block_origin >= block_len) {
          ++block;
          index = params->origin + block * params->stride;
          block_origin = params->origin + block * params->stride;
          terminal_block = block == params->max_blocks - 1;
          block_len =
            (terminal_block ? params->terminal_block_len : params->block_len);
        }
        at_data = outer_at_data && (index - block_origin < block_len);
        at_end =
          terminal_block
          && (index - block_origin == params->max_terminal_block_len);
        --n;
      }
    }

    std::size_t
    num_remaining() const {
      if (at_end || !at_data)
        return 0;
      auto block_origin = params->origin + block * params->stride;
      return
        params->true_length()
        - (block * params->block_len + index - block_origin);
    }

    void
    complete() {
      at_data = false;
      at_end = true;
    }
  };

  static Reader
  begin(
    const std::string& path,
    ::MPI_Comm comm,
    ::MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    bool ms_buffer_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool readahead,
    bool debug_log = false);

  static Reader
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
      && get() == other.get());
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
    mpi_call(::MPI_Comm_size, handles->comm, reinterpret_cast<int*>(&result));
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

  // NB: the returned array can be empty! Caller can test MSArray::buffer
  // value or use MSArray::num_elements().
  const std::shared_ptr<const std::complex<float> >
  get() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    wait_for_array(m_ms_array);
    return std::get<0>(m_ms_array).buffer;
  };

  const std::vector<IndexBlockSequence<MSColumns> >&
  indices() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    wait_for_array(m_ms_array);
    return std::get<0>(m_ms_array).blocks;
  };

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

  const std::shared_ptr<const std::complex<float> >
  operator*() const {
    return get();
  }

protected:

  void
  step(bool cont);

  void
  start_next();

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
    int rank,
    bool debug_log);

  static std::tuple<std::unique_ptr<::MPI_Datatype, DatatypeDeleter>, unsigned>
    init_array_datatype(
      const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
      const std::shared_ptr<std::vector<IterParams> >& iter_params,
      bool ms_buffer_order,
      bool tail_array,
      int rank,
      bool debug_log);

  static std::unique_ptr<::MPI_Datatype, DatatypeDeleter>
    init_fileview(
      const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
      const std::shared_ptr<std::vector<IterParams> >& iter_params,
      const std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
      bool tail_fileview,
      int rank,
      bool debug_log);

  static std::unique_ptr<std::vector<IndexBlockSequenceMap<MSColumns> > >
  make_index_block_sequences(
    const std::shared_ptr<const std::vector<IterParams> >& iter_params);

  std::shared_ptr<const ::MPI_Datatype>
  init_buffer_datatype() const;

  std::tuple<
    std::unique_ptr<std::complex<float> >,
    std::variant<::MPI_Request, ::MPI_Status> >
  read_arrays(
    TraversalState& traversal_state,
    bool nonblocking,
    const std::vector<IndexBlockSequence<MSColumns> >& blocks,
    ::MPI_File file) const;

  void
  set_fileview(TraversalState& traversal_state, ::MPI_File file) const;

  void
  advance_to_next_buffer(TraversalState& traversal_state, ::MPI_File file);

  void
  advance_to_buffer_end(TraversalState& traversal_state);

  std::tuple<MSArray, std::variant<::MPI_Request, ::MPI_Status> >
  read_next_buffer(
    TraversalState& TraversalState,
    bool nonblocking,
    ::MPI_File file);

  mutable std::shared_ptr<const ::MPI_Datatype> m_buffer_datatype;

  bool
  buffer_order_compare(const MSColumns& col0, const MSColumns& col1);

  static const IterParams*
  find_iter_params(
    const std::shared_ptr<std::vector<IterParams> >& iter_params,
    MSColumns col) {

    auto ip =
      std::find_if(
        std::begin(*iter_params),
        std::end(*iter_params),
        [&col](auto& ip) {
          return ip.axis == col;
        });
    if (ip != std::end(*iter_params))
      return &*ip;
    return nullptr;
  }

  void
  wait_for_array(
    std::tuple<MSArray, std::variant<::MPI_Request, ::MPI_Status> >& array,
    bool cancel=false)
    const {
    if (std::holds_alternative<::MPI_Request>(std::get<1>(array))) {
      if (cancel)
        mpi_call(
          ::MPI_Cancel,
          &std::get<::MPI_Request>(std::get<1>(m_ms_array)));

      std::tuple<
        MSArray,
        std::variant<::MPI_Request, ::MPI_Status> > completion(
          std::get<0>(array),
          std::variant<::MPI_Request, ::MPI_Status>(
            std::in_place_index<1>));
      mpi_call(
        ::MPI_Wait,
        &std::get<::MPI_Request>(std::get<1>(array)),
        &std::get<::MPI_Status>(std::get<1>(completion)));
      array = std::move(completion);
    }
  }

private:

  ReaderMPIState m_mpi_state;

  std::shared_ptr<const std::vector<ColumnAxisBase<MSColumns> > > m_ms_shape;

  // ms array iteration definition, for this rank, in traversal order
  std::shared_ptr<const std::vector<IterParams> > m_iter_params;

  std::shared_ptr<const std::vector<MSColumns> > m_buffer_order;

  std::shared_ptr<const std::optional<MSColumns> > m_inner_fileview_axis;

  std::shared_ptr<const ::MPI_Datatype> m_etype_datatype;

  std::shared_ptr<const ArrayIndexer<MSColumns> > m_ms_indexer;

  int m_rank;

  std::size_t m_buffer_size;

  bool m_readahead;

  bool m_debug_log;

  TraversalState m_traversal_state;

  TraversalState m_next_traversal_state;

  mutable std::tuple<MSArray, std::variant<::MPI_Request, ::MPI_Status> >
  m_ms_array;

  std::tuple<MSArray, std::variant<::MPI_Request, ::MPI_Status> >
  m_next_ms_array;

  mutable std::mutex m_mtx;
};

} // end namespace mpims

#endif // READER_H_
