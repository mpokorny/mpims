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

namespace mpims {

enum AMode {
  WriteOnly,
  ReadWrite
};

class Reader
  : public std::iterator<std::input_iterator_tag, const MSArray, std::size_t> {

  friend class Writer;

public:

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

  class IterParams;
  class AxisIter;

  // TraversalState maintains the state used to traverse the MS in the order
  // provided by the client. If one considers MS traversal as stack of
  // traversals along every axis with recursion to get to the next axis, this
  // structure contains the information needed for traversal after flattening
  // such a recursive iteration into a single loop.
  struct TraversalState {

    TraversalState()
      : cont(false)
      , global_eof(true) {
    }

    TraversalState(
      const std::shared_ptr<const std::vector<IterParams> >& iter_params_,
      MSColumns outer_ms_axis,
      std::size_t outer_ms_length,
      bool in_tail_,
      MSColumns outer_full_array_axis_,
      const std::shared_ptr<const MPI_Datatype>& full_buffer_datatype,
      unsigned full_buffer_dt_count,
      const std::shared_ptr<const MPI_Datatype>& tail_buffer_datatype,
      unsigned tail_buffer_dt_count,
      const std::shared_ptr<const MPI_Datatype>& full_fileview_datatype,
      const std::shared_ptr<const MPI_Datatype>& tail_fileview_datatype)
    : iter_params(iter_params_)
    , block_maps(make_index_block_sequences(iter_params_))
    , count(0)
    , max_count(0)
    , in_tail(in_tail_)
    , outer_full_array_axis(outer_full_array_axis_)
    , m_full_buffer_datatype(full_buffer_datatype)
    , m_full_buffer_dt_count(full_buffer_dt_count)
    , m_tail_buffer_datatype(tail_buffer_datatype)
    , m_tail_buffer_dt_count(tail_buffer_dt_count)
    , m_full_fileview_datatype(full_fileview_datatype)
    , m_tail_fileview_datatype(tail_fileview_datatype) {

      global_eof = false;
      const IterParams* init_params = &(*iter_params)[0];
      axis_iters.emplace_back(
        std::shared_ptr<const IterParams>(iter_params, init_params),
        !init_params->max_blocks || init_params->max_blocks > 0);
      std::for_each(
        std::begin(*iter_params),
        std::end(*iter_params),
        [this](const IterParams& ip) {
          data_index[ip.axis] = ip.origin;
        });

      if (axis_iters.front().params->axis == outer_ms_axis) {
        eof_axis_iters.push_back(axis_iters.front());
        auto& eai = eof_axis_iters.front();
        max_count = std::max(eai.params->buffer_capacity, 1uL);
        while (!eai.at_end && eai.index < outer_ms_length)
          eai.increment();
      }
      cont = true;
    }

    //EOF condition flag
    bool
    eof() const {
      if (axis_iters.empty() || global_eof)
        return true;
      if (eof_axis_iters.empty())
        return false;
      auto ai = std::rbegin(axis_iters);
      auto ai_end = std::rend(axis_iters);
      auto eai = std::rbegin(eof_axis_iters);
      auto eai_end = std::rend(eof_axis_iters);
      bool result = false;
      auto num_undefined = axis_iters.size() - eof_axis_iters.size();
      while (num_undefined-- > 0 && ai != ai_end)
        ++ai;
      while (!result && ai != ai_end && eai != eai_end) {
        result = ai->index >= eai->index;
        ++ai;
        ++eai;
      }
      return result;
    };

    bool
    at_end() const {
      return !cont || global_eof;
    }

    // continuation condition flag (to allow breaking out from iteration)
    bool cont;

    bool global_eof;

    std::shared_ptr<const std::vector<IterParams> > iter_params;

    std::shared_ptr<
      const std::vector<IndexBlockSequenceMap<MSColumns> > > block_maps;

    ArrayIndexer<MSColumns>::index data_index;

    // stack of AxisIters to maintain axis iteration indexes
    std::deque<AxisIter> axis_iters;

    std::deque<AxisIter> eof_axis_iters;

    int count;

    int max_count;

    // Is traversal currently in a tail fileview? A tail fileview differs
    // somewhat from a non-tail fileview when the iteration at the outermost
    // fileview axis doesn't fit into the same pattern at the end of the axis as
    // all the previous fileviews. This can happen when the distribution of data
    // on an axis to processes is uneven at the end of an axis.
    bool in_tail;

    // the outermost axis at which data can fully fit into a buffer
    MSColumns outer_full_array_axis;

    std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned>
    buffer_datatype() const {
      if (in_tail)
        return std::make_tuple(m_tail_buffer_datatype, m_tail_buffer_dt_count);
      else
        return std::make_tuple(m_full_buffer_datatype, m_full_buffer_dt_count);
    }

    std::shared_ptr<const MPI_Datatype>
    fileview_datatype() const {
      return (in_tail ? m_tail_fileview_datatype : m_full_fileview_datatype);
    }

    bool
    operator==(const TraversalState& other) const {
      return (
        cont == other.cont
        && count == other.count
        && (block_maps == other.block_maps || *block_maps == *other.block_maps)
        && data_index == other.data_index
        && axis_iters == other.axis_iters
        && eof_axis_iters == other.eof_axis_iters
        && in_tail == other.in_tail
        && m_tail_buffer_dt_count == other.m_tail_buffer_dt_count
        && m_full_buffer_dt_count == other.m_full_buffer_dt_count
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
    blocks(int cnt) const {
      std::vector<IndexBlockSequence<MSColumns> > result;
      if (cnt > 0) {
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

    void
    advance_to_buffer_end() {
      if (!axis_iters.empty()) {
        AxisIter* axis_iter = &axis_iters.back();
        axis_iter->increment(max_count);
        data_index[axis_iter->params->axis] = axis_iter->index;
        while (!axis_iters.empty() && axis_iter->at_end) {
          data_index[axis_iter->params->axis] = axis_iter->params->origin;
          axis_iters.pop_back();
          if (!axis_iters.empty()) {
            axis_iter = &axis_iters.back();
            axis_iter->increment();
            data_index[axis_iter->params->axis] = axis_iter->index;
          }
        }}
    }

    template <typename F>
    void
    advance_to_next_buffer(
      std::shared_ptr<const std::optional<MSColumns> > inner_fileview_axis,
      F at_fileview_axis) {

      count = 0;
      max_count = 0;
      while (!axis_iters.empty()) {
        AxisIter& axis_iter = axis_iters.back();
        MSColumns axis = axis_iter.params->axis;
        if (!axis_iter.at_end) {
          auto depth = axis_iters.size();
          data_index[axis] = axis_iter.index;
          if (axis_iter.params->buffer_capacity > 0) {
            max_count = static_cast<int>(axis_iter.params->buffer_capacity);
            auto nr = axis_iter.num_remaining();
            if (nr && (nr.value() < static_cast<std::size_t>(max_count))) {
              count = nr.value();
              in_tail = true;
            } else {
              count = max_count;
              in_tail = false;
            }
          } else {
            in_tail = false;
          }
          if (*inner_fileview_axis && axis == inner_fileview_axis->value())
            at_fileview_axis(axis);
          if (axis_iter.params->buffer_capacity > 0)
            return;
          const IterParams* next_params = &(*iter_params)[depth];
          axis_iters.emplace_back(
            std::shared_ptr<const IterParams>(iter_params, next_params),
            axis_iter.at_data);
        } else {
          data_index[axis] = axis_iter.params->origin;
          axis_iters.pop_back();
          if (!axis_iters.empty())
            axis_iters.back().increment();
        }
      }
    }

  private:

    std::shared_ptr<const MPI_Datatype> m_full_buffer_datatype;

    unsigned m_full_buffer_dt_count;

    std::shared_ptr<const MPI_Datatype> m_tail_buffer_datatype;

    unsigned m_tail_buffer_dt_count;

    std::shared_ptr<const MPI_Datatype> m_full_fileview_datatype;

    std::shared_ptr<const MPI_Datatype> m_tail_fileview_datatype;

    std::ostringstream
    show_iters(const std::deque<AxisIter>& iters) const {
      std::ostringstream result;
      std::for_each(
        std::begin(iters),
        std::end(iters),
        [&result](const auto& ai) {
          result << mscol_nickname(ai.params->axis) << ":" << ai.index << ";";
        });
      return result;
    }
  };

  Reader();

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

  // IterParams holds the information for the iteration over a single axis, as
  // well as context within the scope of iteration over the entire MS together
  // given a user buffer size. Most of the values in this structure are used to
  // describe the iteration over an axis for a particular process.
  struct IterParams {
    MSColumns axis;
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
    std::size_t origin, stride, block_len,
      terminal_block_len, max_terminal_block_len;
    std::optional<std::size_t> length, max_blocks;

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

    std::optional<std::size_t>
    accessible_length() const {
      if (max_blocks.has_value())
        return block_len * (max_blocks.value() - 1) + terminal_block_len;
      else
        return std::nullopt;
    }

    std::optional<std::size_t>
    max_accessible_length() const {
      if (max_blocks.has_value())
        return block_len * (max_blocks.value() - 1) + max_terminal_block_len;
      else
        return std::nullopt;
    }
  };

  // AxisIter contains the state of the iteration along an axis. It follows the
  // iteration pattern defined for an axis by an IterParams value.
  class AxisIter {
  public:

    AxisIter(std::shared_ptr<const IterParams> params_, bool outer_at_data_)
      : params(params_)
      , index(params_->origin)
      , block(0)
      , at_data((!params_->max_blocks || params_->max_blocks.value() > 0)
                && outer_at_data_)
      , outer_at_data(outer_at_data_)
      , at_end(params_->max_blocks && params_->max_blocks.value() == 0) {
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
        bool terminal_block =
          (params->max_blocks
           ? (block == params->max_blocks.value() - 1)
           : false);
        auto block_len =
          (terminal_block ? params->terminal_block_len : params->block_len);
        ++index;
        if (!terminal_block && index - block_origin >= block_len) {
          ++block;
          index = params->origin + block * params->stride;
          block_origin = index;
          terminal_block = (
            params->max_blocks
            ? (block == params->max_blocks.value() - 1)
            : false);
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

    std::optional<std::size_t>
    num_remaining() const {
      if (at_end || !at_data)
        return 0;
      auto accessible_length = params->accessible_length();
      if (accessible_length) {
        auto block_origin = params->origin + block * params->stride;
        return (accessible_length.value()
                - (block * params->block_len + index - block_origin));
      } else {
        return std::nullopt;
      }
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

  static std::tuple<std::unique_ptr<MPI_Datatype, DatatypeDeleter>, unsigned>
  init_buffer_datatype(
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::shared_ptr<std::vector<IterParams> >& iter_params,
    bool ms_buffer_order,
    bool tail_array,
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

  static std::unique_ptr<MPI_Datatype, DatatypeDeleter>
  init_fileview(
    MPI_Aint value_extent,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::shared_ptr<std::vector<IterParams> >& iter_params,
    const std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
    bool tail_fileview,
    int rank,
    bool debug_log);

  static std::unique_ptr<std::vector<IndexBlockSequenceMap<MSColumns> > >
  make_index_block_sequences(
    const std::shared_ptr<const std::vector<IterParams> >& iter_params);

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

  static const IterParams*
  find_iter_params(
    const std::shared_ptr<const std::vector<IterParams> >& iter_params,
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
