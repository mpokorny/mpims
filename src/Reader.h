/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef READER_H_
#define READER_H_

#include <mpi.h>

#include <array>
#include <cassert>
#include <complex>
#include <stack>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <ColumnAxis.h>
#include <IndexBlock.h>
#include <MSColumns.h>
#include <DataDistribution.h>
#include <ReaderMPIState.h>

namespace mpims {

class Reader {

  // define a few helper classes up front

  struct IterParams {
    MSColumns axis;
    bool in_array, within_fileview;
    std::size_t length, origin, stride, block_len, max_blocks,
      terminal_block_len, max_terminal_block_len;
  };

  class AxisIter {
  public:

    AxisIter(const IterParams& params_, bool outer_at_data_)
      : params(params_)
      , index(params_.origin)
      , at_data(params_.max_blocks > 0 && outer_at_data_)
      , outer_at_data(outer_at_data_)
      , at_end(params_.max_blocks == 0) {
    }

    const IterParams& params;
    std::size_t index;
    std::size_t block;
    bool at_data;
    bool outer_at_data;
    bool at_end;

    void
    increment() {
      if (!at_end) {
        block = index / params.stride;
        auto block_origin = params.origin + block * params.stride;
        bool terminal_block = block == params.max_blocks - 1;
        auto block_len =
          (terminal_block ? params.terminal_block_len : params.block_len);
        ++index;
        if (!terminal_block && index - block_origin >= block_len) {
          ++block;
          index = params.origin + block * params.stride;
          block_origin = params.origin + block * params.stride;
          terminal_block = block == params.max_blocks - 1;
          block_len =
            (terminal_block ? params.terminal_block_len : params.block_len);
        }
        at_data = outer_at_data && (index - block_origin < block_len);
        at_end =
          terminal_block
          && (index - block_origin == params.max_terminal_block_len);
      }
    }

    void
    complete() {
      at_data = false;
      at_end = true;
    }
  };

  class TraversalState {
  public:

    TraversalState(
      std::vector<IndexBlockSequence<MSColumns> >&& indexes_,
      std::vector<IterParams>& iter_params)
      : eof(false)
      , cont(true)
      , indexes(std::move(indexes_)) {
      std::for_each(
        std::begin(iter_params),
        std::end(iter_params),
        [this](const IterParams& ip) {
          data_index[ip.axis] = ip.origin;
        });
    };

    bool eof;
    bool cont;
    std::vector<IndexBlockSequence<MSColumns> > indexes;
    ArrayIndexer<MSColumns>::index data_index;
    std::stack<AxisIter> axis_iters;
  };

public:

  Reader(
    const std::string& path,
    ::MPI_Comm comm,
    ::MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool debug_log = false);

  template <typename F>
  void
  iterate(const F& callback) {
    auto handles = m_mpi_state.handles();
    std::lock_guard<MPIHandles> lock(*handles);
    if (handles->comm != MPI_COMM_NULL)
      loop(callback);
  }

private:

  std::vector<ColumnAxisBase<MSColumns> > m_ms_shape;

  ReaderMPIState m_mpi_state;

  std::shared_ptr<::MPI_Datatype> m_array_datatype;

  int m_rank;

  std::size_t m_buffer_size;

  // ms array iteration definition, for this rank, in traversal order
  std::vector<IterParams> m_iter_params;

  MSColumns m_outer_array_axis;

  std::optional<MSColumns> m_inner_fileview_axis;

  std::shared_ptr<::MPI_Datatype> m_fileview_datatype;

  std::shared_ptr<ArrayIndexer<MSColumns> > m_ms_indexer;

  bool m_debug_log;

  void
  init_iterparams(
    const std::vector<MSColumns>& traversal_order,
    const std::unordered_map<MSColumns, DataDistribution>& pgrid);

  void
  init_traversal_partitions();

  void
  init_array_datatype();

  void
  init_fileview();

  void
  set_fileview(ArrayIndexer<MSColumns>::index& index);

  std::vector<IndexBlockSequence<MSColumns> >
  make_index_block_sequences();

  std::shared_ptr<std::complex<float> >
  read_array(bool at_data);

  TraversalState
  begin();

  template <typename F>
  void
  next(TraversalState& state, const F& callback) {
    assert(!end(state));
    auto depth = state.axis_iters.size();
    AxisIter& axis_iter = state.axis_iters.top();
    const MSColumns& axis = axis_iter.params.axis;
    if (!axis_iter.at_end) {
      state.indexes[depth - 1].m_blocks[0].m_index = axis_iter.index;
      state.data_index[axis] = axis_iter.index;
      if (m_inner_fileview_axis && axis == m_inner_fileview_axis.value())
        set_fileview(state.data_index);
      if (axis_iter.params.in_array) {

        if (m_debug_log) {
          std::ostringstream oss;
          auto data_index = state.data_index;
          oss << "(" << m_rank << ") read";
          std::for_each(
            std::begin(m_iter_params),
            std::end(m_iter_params),
            [&data_index, &oss](const IterParams& ip) {
              oss << "; " << mscol_nickname(ip.axis)
                  << " " << data_index[ip.axis];
            });
          oss << std::endl;
          std::clog << oss.str();
        }

        std::shared_ptr<std::complex<float> > buffer(
          read_array(axis_iter.at_data));
        if (buffer) {
          state.cont = callback(state.indexes, buffer);
          state.eof = false;
        } else {
          state.cont = true;
          state.eof = true;
        }
        std::array<bool, 2> tests { state.cont, state.eof };
        {
          auto handles = m_mpi_state.handles();
          std::lock_guard<MPIHandles> lock(*handles);
          mpi_call(
            ::MPI_Allreduce,
            MPI_IN_PLACE,
            tests.data(),
            tests.size(),
            MPI_CXX_BOOL,
            MPI_LAND,
            handles->comm);
        }
        state.cont = tests[0];
        state.eof = tests[1];
        axis_iter.complete();
      } else {
        const IterParams& next_params = m_iter_params[depth];
        state.axis_iters.emplace(next_params, axis_iter.at_data);
      }
    } else {
      state.data_index[axis] = axis_iter.params.origin;
      state.axis_iters.pop();
      if (!state.axis_iters.empty())
        state.axis_iters.top().increment();
    }
  }

  bool
  end(TraversalState& state);

  template <typename F>
  void
  loop(const F& callback) {

    TraversalState state = begin();
    while (!end(state))
      next(state, callback);
  }
};

} // end namespace mpims

#endif // READER_H_
