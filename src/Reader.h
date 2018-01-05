/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef READER_H_
#define READER_H_

#include <mpi.h>

#include <complex>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <ColumnAxis.h>
#include <IndexBlock.h>
#include <MSColumns.h>
#include <ProcessDistribution.h>

namespace mpims {

class Reader {
public:

  Reader(
    const std::string& path,
    ::MPI_Comm comm,
    ::MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    std::unordered_map<MSColumns, ProcessDistribution>& pgrid,
    std::size_t max_buffer_size);

  virtual ~Reader() {
    finalize();
  };

  void
  finalize();

  template <typename F>
  void
  iterate(const F& callback) {
    loop(callback);
  }

private:

  std::vector<ColumnAxisBase<MSColumns> > m_ms_shape;

  ::MPI_Comm m_comm;

  ::MPI_File m_file;

  bool m_array_datatype_predef;

  ::MPI_Datatype m_array_datatype;

  int m_rank;

  std::size_t m_buffer_size;

  struct IterParams {
    MSColumns axis;
    bool in_array;
    std::size_t length, origin, stride, block_len, max_blocks;
  };

  // ms array iteration definition, for this rank, in traversal order
  std::vector<IterParams> m_iter_params;

  MSColumns m_outer_array_axis;

  MSColumns m_inner_fileview_axis;

  bool m_fileview_datatype_predef;

  ::MPI_Datatype m_fileview_datatype;

  std::shared_ptr<ArrayIndexer<MSColumns> > m_ms_indexer;

  class AxisIter {
  public:

    AxisIter(const IterParams& params, bool outer_at_data)
      : m_params(params)
      , m_index(0)
      , m_at_data(params.max_blocks > 0 && outer_at_data)
      , m_outer_at_data(outer_at_data)
      , m_at_end(params.max_blocks == 0) {
    }

    const IterParams& m_params;
    std::size_t m_index;
    bool m_at_data;
    bool m_outer_at_data;
    bool m_at_end;

    void
    increment() {
      if (!m_at_end) {
        auto block = (m_index - m_params.origin) / m_params.stride;
        auto block_origin = m_params.origin + block * m_params.stride;
        ++m_index;
        if (m_index - block_origin >= m_params.block_len) {
          ++block;
          m_index = m_params.origin + block * m_params.stride;
        }
        m_at_data = m_outer_at_data && m_index < m_params.length;
        m_at_end = block == m_params.max_blocks;
      }
    }

    void
    complete() {
      m_at_data = false;
      m_at_end = true;
    }
  };

  void
  init_iterparams(
    const std::vector<MSColumns>& traversal_order,
    const std::unordered_map<MSColumns, ProcessDistribution>& pgrid);

  void
  init_outer_array_axis();

  void
  init_array_datatype();

  void
  init_fileview();

  void
  set_fileview(ArrayIndexer<MSColumns>::index& index);

  std::vector<IndexBlockSequence<MSColumns> >
  make_index_block_sequences();

  std::shared_ptr<std::complex<float> >
  read_array();

  template <typename F>
  void
  loop(const F& callback) {

    bool eof = false;
    std::vector<IndexBlockSequence<MSColumns> > indexes =
      make_index_block_sequences();
    ArrayIndexer<MSColumns>::index data_index;
    std::stack<AxisIter> axis_iters;
    axis_iters.emplace(m_iter_params[0], m_iter_params[0].max_blocks > 0);
    while (!(eof || axis_iters.empty())) {
      auto depth = axis_iters.size();
      AxisIter& axis_iter = axis_iters.top();
      if (!axis_iter.m_at_end) {
        const MSColumns& axis = axis_iter.m_params.axis;
        indexes[depth - 1].m_blocks[0].m_index = axis_iter.m_index;
        data_index[axis] = axis_iter.m_index;
        if (axis == m_inner_fileview_axis)
          set_fileview(data_index);
        if (axis == m_outer_array_axis) {
          eof = true;
          std::shared_ptr<std::complex<float> > buffer;
          if (axis_iter.m_at_data) {
            buffer = read_array();
            if (buffer) {
              callback(indexes, buffer);
              eof = false;
            }
          }
          mpi_call(
            ::MPI_Allreduce,
            MPI_IN_PLACE,
            &eof,
            1,
            MPI_CXX_BOOL,
            MPI_LAND,
            m_comm);
          axis_iter.complete();
        } else {
          const IterParams& next_params = m_iter_params[depth];
          axis_iters.emplace(next_params, axis_iter.m_at_data);
        }
      } else {
        axis_iters.pop();
        if (!axis_iters.empty())
          axis_iters.top().increment();
      }
    }
  }
};

} // end namespace mpims

#endif // READER_H_
