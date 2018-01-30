/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <Reader.h>

using namespace mpims;

Reader
Reader::begin(
  const std::string& path,
  ::MPI_Comm comm,
  ::MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t max_buffer_size,
  bool debug_log) {

  std::size_t buffer_size =
    (max_buffer_size / sizeof(std::complex<float>))
    * sizeof(std::complex<float>);

  // reduce process grid for small (blocked) axis sizes
  for (auto& pg : pgrid) {
    MSColumns pcol;
    DataDistribution pdist;
    std::tie(pcol, pdist) = pg;
    auto ax =
      std::find_if(
        std::begin(ms_shape),
        std::end(ms_shape),
        [&pcol](const ColumnAxisBase<MSColumns>& ax) {
          return ax.id() == pcol;
        });
    pdist.block_size = std::min(ax->length(), pdist.block_size);
    pdist.num_processes =
      std::min(
        (ax->length() + pdist.block_size - 1) / pdist.block_size,
        pdist.num_processes);
  }

  // compute process grid size
  std::size_t pgrid_size = 1;
  std::for_each(
    std::begin(traversal_order),
    std::end(traversal_order),
    [&pgrid_size, &pgrid]
    (const MSColumns& col) mutable {
      auto np = pgrid.find(col);
      if (np != std::end(pgrid))
        pgrid_size *= pgrid[col].num_processes;
    });
  int comm_size;
  mpi_call(::MPI_Comm_size, comm, &comm_size);
  if (static_cast<std::size_t>(comm_size) < pgrid_size)
    throw std::runtime_error("too few processes for grid");

  // create a new communicator of the minimum needed size
  ::MPI_Comm reduced_comm;
  if (static_cast<std::size_t>(comm_size) > pgrid_size) {
    int comm_rank;
    mpi_call(::MPI_Comm_rank, comm, &comm_rank);
    mpi_call(
      ::MPI_Comm_split,
      comm,
      ((static_cast<std::size_t>(comm_rank) < pgrid_size) ? 1 : MPI_UNDEFINED),
      comm_rank,
      &reduced_comm);
  } else {
    mpi_call(::MPI_Comm_dup, comm, &reduced_comm);
  }
  int rank = 0;
  if (reduced_comm != MPI_COMM_NULL) {
    mpi_call(::MPI_Comm_set_errhandler, reduced_comm, MPI_ERRORS_RETURN);
    mpi_call(::MPI_Comm_rank, reduced_comm, &rank);
  }

  std::shared_ptr<std::vector<IterParams> > iter_params =
    std::make_shared<std::vector<IterParams> >(
      traversal_order.size());
  std::shared_ptr<std::optional<MSColumns> > inner_fileview_axis =
    std::make_shared<std::optional<MSColumns> >();
  std::shared_ptr<ArrayIndexer<MSColumns> > ms_indexer =
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, ms_shape);
  std::stack<AxisIter> traversal_axis_iters;

  init_iterparams(
    ms_shape,
    traversal_order,
    pgrid,
    rank,
    debug_log,
    iter_params);

  init_traversal_partitions(
    ms_shape,
    buffer_size,
    iter_params,
    inner_fileview_axis,
    rank,
    debug_log);
  auto ip = std::begin(*iter_params);
  while (ip->buffer_capacity == 0) ++ip;
  auto array_read = ip;
  while (!ip->fully_in_array) ++ip;
  MSColumns outer_full_array_axis = ip->axis;
  if (debug_log) {
    std::clog << "outer full array axis "
              << mscol_nickname(outer_full_array_axis)
              << std::endl;
    std::clog << "read " << array_read->buffer_capacity
              << " arrays at " << mscol_nickname(array_read->axis)
              << std::endl;
    if (inner_fileview_axis->has_value())
      std::clog << "m_inner_fileview_axis "
                << mscol_nickname(inner_fileview_axis->value())
                << std::endl;
    else
      std::clog << "no inner_fileview_axis"
                << std::endl;
  }

  std::shared_ptr<::MPI_Datatype> full_fileview_datatype =
    init_fileview(
      ms_shape,
      iter_params,
      ms_indexer,
      false,
      rank,
      debug_log);

  std::shared_ptr<::MPI_Datatype> tail_fileview_datatype =
    init_fileview(
      ms_shape,
      iter_params,
      ms_indexer,
      true,
      rank,
      debug_log);
  if (!tail_fileview_datatype)
    tail_fileview_datatype = full_fileview_datatype;

  std::shared_ptr<::MPI_Datatype> full_array_datatype =
    init_array_datatype(
      ms_shape,
      iter_params,
      false,
      rank,
      debug_log);

  std::shared_ptr<::MPI_Datatype> tail_array_datatype =
    init_array_datatype(
      ms_shape,
      iter_params,
      true,
      rank,
      debug_log);
  if (!tail_array_datatype)
    tail_array_datatype = full_array_datatype;

  ::MPI_File file = MPI_FILE_NULL;
  ::MPI_Info priv_info = info;
  if (info != MPI_INFO_NULL)
    mpi_call(::MPI_Info_dup, info, &priv_info);
  if (reduced_comm != MPI_COMM_NULL) {
    mpi_call(
      ::MPI_File_open,
      reduced_comm,
      path.c_str(),
      MPI_MODE_RDONLY,
      priv_info,
      &file);
    mpi_call(::MPI_File_set_errhandler, file, MPI_ERRORS_RETURN);
  }

  TraversalState traversal_state(
    reduced_comm,
    iter_params,
    outer_full_array_axis,
    full_array_datatype,
    tail_array_datatype,
    full_fileview_datatype,
    tail_fileview_datatype);

  std::for_each(
    std::begin(*iter_params),
    std::end(*iter_params),
    [&traversal_state](const IterParams& ip) {
      traversal_state.data_index[ip.axis] = ip.origin;
    });

  Reader result =
    Reader(
      ReaderMPIState(reduced_comm, priv_info, file, path),
      std::make_shared<std::vector<ColumnAxisBase<MSColumns> > >(ms_shape),
      iter_params,
      inner_fileview_axis,
      ms_indexer,
      rank,
      buffer_size,
      std::move(traversal_state),
      debug_log);

  auto handles = result.m_mpi_state.handles();
  // don't need to acquire handles lock here, since "result" is local
  if (handles->file != MPI_FILE_NULL) {
    if (!*result.m_inner_fileview_axis)
      result.set_fileview(handles->file);
    const IterParams* init_params = &(*result.m_iter_params)[0];
    result.m_traversal_state.axis_iters.emplace(
      std::shared_ptr<const IterParams>(result.m_iter_params, init_params),
      init_params->max_blocks > 0);
    result.read_next_buffer(handles->file);
  } else {
    result.m_traversal_state.eof = true;
  }

  return result;
}

void
Reader::step(bool cont) {
  std::lock_guard<decltype(m_mtx)> lock(m_mtx);

  if (at_end())
    throw std::out_of_range("Reader cannot be advanced: at end");

  advance_to_buffer_end();

  m_traversal_state.cont = cont;
  m_traversal_state.eof = m_traversal_state.axis_iters.empty();
  std::array<bool, 2> tests { m_traversal_state.cont, m_traversal_state.eof };
  auto handles = m_mpi_state.handles();
  std::lock_guard<MPIHandles> lock1(*handles);
  mpi_call(
    ::MPI_Allreduce,
    MPI_IN_PLACE,
    tests.data(),
    tests.size(),
    MPI_CXX_BOOL,
    MPI_LAND,
    handles->comm);
  m_ms_array = MSArray();
  if (m_traversal_state.cont && !m_traversal_state.eof)
    read_next_buffer(handles->file);
}

bool
Reader::at_end() const {
  return !m_traversal_state.cont
    || m_traversal_state.eof
    || m_traversal_state.axis_iters.empty();
}

void
Reader::swap(Reader& other) {
  using std::swap;
  std::lock_guard<decltype(m_mtx)> lock1(m_mtx);
  std::lock_guard<decltype(other.m_mtx)> lock2(other.m_mtx);
  swap(m_ms_shape, other.m_ms_shape);
  swap(m_mpi_state, other.m_mpi_state);
  swap(m_rank, other.m_rank);
  swap(m_buffer_size, other.m_buffer_size);
  swap(m_iter_params, other.m_iter_params);
  swap(m_etype_datatype, other.m_etype_datatype);
  swap(m_inner_fileview_axis, other.m_inner_fileview_axis);
  swap(m_ms_indexer, other.m_ms_indexer);
  swap(m_debug_log, other.m_debug_log);
  swap(m_traversal_state, other.m_traversal_state);
}

void
Reader::init_iterparams(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  const std::unordered_map<MSColumns, DataDistribution>& pgrid,
  int rank,
  bool debug_log,
  std::shared_ptr<std::vector<IterParams> >& iter_params) {

  std::size_t dist_size = 1;
  std::size_t array_length = 1;
  std::for_each(
    std::begin(ms_shape),
    std::end(ms_shape),
    [&] (const ColumnAxisBase<MSColumns>& ax) {
      // find index of this column in traversal_order
      MSColumns col = ax.id();
      std::size_t length = ax.length();
      std::size_t traversal_index = 0;
      while (traversal_order[traversal_index] != col)
        ++traversal_index;

      // create IterParams for this axis
      std::size_t num_processes;
      std::size_t block_size;
      if (pgrid.count(col) > 0) {
        auto pg = pgrid.at(col);
        num_processes = pg.num_processes;
        block_size = pg.block_size;
      } else {
        num_processes = 1;
        block_size = 1;
      }
      std::size_t grid_len = num_processes;
      std::size_t order = (rank / dist_size) % grid_len;
      std::size_t block_len = block_size;
      assert(block_len <= length);
      std::size_t origin = order * block_len;
      std::size_t stride = grid_len * block_len;
      std::size_t max_blocks = (length + stride - 1) / stride;
      std::size_t blocked_rem =
        ((length + block_len - 1) / block_len) % grid_len;
      std::size_t terminal_block_len;
      std::size_t max_terminal_block_len;
      if (blocked_rem == 0) {
        terminal_block_len = block_len;
        max_terminal_block_len = block_len;
      } else if (blocked_rem == 1) {
        terminal_block_len = ((order == 0) ? (length % block_len) : 0);
        max_terminal_block_len = length % block_len;
      } else {
        terminal_block_len =
          ((order < blocked_rem - 1)
           ? block_len
           : ((order == blocked_rem - 1) ? (length % block_len) : 0));
        max_terminal_block_len = block_len;
      }
      (*iter_params)[traversal_index] =
        IterParams { col, false, false, 0, array_length,
                     length, origin, stride, block_len,
                     max_blocks, terminal_block_len, max_terminal_block_len };
      if (debug_log) {
        std::clog << "(" << rank << ") "
                  << mscol_nickname(col)
                  << " length: "
                  << (*iter_params)[traversal_index].length
                  << ", origin: "
                  << (*iter_params)[traversal_index].origin
                  << ", stride: "
                  << (*iter_params)[traversal_index].stride
                  << ", block_len: "
                  << (*iter_params)[traversal_index].block_len
                  << ", max_blocks: "
                  << (*iter_params)[traversal_index].max_blocks
                  << ", terminal_block_len: "
                  << (*iter_params)[traversal_index].terminal_block_len
                  << ", max_terminal_block_len: "
                  << (*iter_params)[traversal_index].max_terminal_block_len
                  << std::endl;
      }
      dist_size *= grid_len;
    });
}

void
Reader::init_traversal_partitions(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  std::size_t& buffer_size,
  std::shared_ptr<std::vector<IterParams> >& iter_params,
  std::shared_ptr<std::optional<MSColumns> >& inner_fileview_axis,
  int rank,
  bool debug_log) {
  // compute shape of largest array with full-length dimensions (for this rank)
  // that buffer of requested size can contain, building up from innermost
  // traversal axis...this is the full array partition, as it defines the axis
  // at which the data occupies a single array in memory
  std::size_t max_buffer_length = buffer_size / sizeof(std::complex<float>);
  if (max_buffer_length == 0)
    throw std::runtime_error("maximum buffer size too small");
  std::size_t array_length = 1;
  auto start_buffer = iter_params->rbegin();
  while (start_buffer != iter_params->rend()) {
    start_buffer->array_length = array_length;
    std::size_t full_len = start_buffer->block_len * start_buffer->max_blocks;
    array_length *= full_len;
    std::size_t len =
      std::min(
        (max_buffer_length / start_buffer->array_length)
        / start_buffer->block_len,
        start_buffer->max_blocks)
      * start_buffer->block_len;
    if (len == 0) {
      if (start_buffer != iter_params->rbegin()) {
        --start_buffer;
        start_buffer->buffer_capacity =
          start_buffer->max_blocks * start_buffer->block_len;
        start_buffer->fully_in_array = false;
        break;
      } else {
        throw std::runtime_error("maximum buffer size too small");
      }
    }
    if (len > 0) {
      start_buffer->buffer_capacity = len;
      if (start_buffer != iter_params->rbegin()) {
        auto prev_buffer = start_buffer - 1;
        prev_buffer->fully_in_array = true;
        prev_buffer->buffer_capacity = 0;
      }
    }
    if (len < full_len)
      break;
    ++start_buffer;
  }
  if (start_buffer == iter_params->rend()) {
    --start_buffer;
    start_buffer->buffer_capacity =
      start_buffer->max_blocks * start_buffer->block_len;
    start_buffer->fully_in_array = false;
  }

  assert(start_buffer->buffer_capacity > 0);
  assert(start_buffer == iter_params->rbegin()
         || (start_buffer - 1)->fully_in_array);
  assert(start_buffer->buffer_capacity * start_buffer->array_length
         <= max_buffer_length);
  buffer_size =
    start_buffer->buffer_capacity
    * start_buffer->array_length
    * sizeof(std::complex<float>);

  if (debug_log) {
    std::clog << "(" << rank << ") ";
    std::for_each(
      std::begin(*iter_params),
      std::end(*iter_params),
      [](auto& ip) {
        std::clog << mscol_nickname(ip.axis)
                  << " capacity " << ip.buffer_capacity
                  << ", fully_in " << ip.fully_in_array
                  << ", array_len " << ip.array_length
                  << std::endl;
      });
  }

  // determine which axes are out of order with respect to the MS order
  std::unordered_set<MSColumns> out_of_order;
  {
    auto ip = iter_params->crbegin();
    auto ms = ms_shape.crbegin();
    while (ms != ms_shape.crend()) {
      if (ms->id() == ip->axis)
        ++ip;
      else
        out_of_order.insert(ms->id());
      ++ms;
    }
  }

  // determine the axis at which the traversal order is incompatible with the MS
  // order, with the knowledge that out of order traversal is OK if the
  // reordering can be done in memory (that, is within the partial array
  // partition)...this is the fileview partition, as it defines the axis at
  // which the fileview must be created
  {
    bool in_array_reordering = false;
    auto ip = iter_params->rbegin();
    while (ip != iter_params->rend()) {
      if (!*inner_fileview_axis) {
        if (out_of_order.count(ip->axis) > 0 || in_array_reordering) {
          if (!(ip->fully_in_array
                || ip->buffer_capacity == ip->block_len * ip->max_blocks))
            *inner_fileview_axis = ip->axis;
          else
            in_array_reordering = true;
        }
      }
      ip->within_fileview = !inner_fileview_axis->has_value();
      ++ip;
    }
  }
}

std::unique_ptr<::MPI_Datatype, DatatypeDeleter>
Reader::init_array_datatype(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::shared_ptr<std::vector<IterParams> >& iter_params,
  bool tail_array,
  int rank,
  bool debug_log) {

  ArrayIndexer<MSColumns>::index index;
  std::for_each(
    std::begin(*iter_params),
    std::end(*iter_params),
    [&index](const IterParams& ip) {
      index[ip.axis] = 0;
    });

  std::size_t buffer_capacity = 0;
  std::size_t tail_buffer_capacity = 0;
  std::vector<ColumnAxisBase<MSColumns> > buffer;
  std::for_each(
    std::begin(*iter_params),
    std::end(*iter_params),
    [&](const IterParams& ip) {
      if (ip.fully_in_array) {
        buffer.emplace_back(static_cast<unsigned>(ip.axis), ip.true_length());
      } else if (ip.buffer_capacity > 0) {
        buffer_capacity = ip.buffer_capacity;
        tail_buffer_capacity = ip.true_length() % ip.buffer_capacity;
        buffer.emplace_back(
          static_cast<unsigned>(ip.axis),
          tail_array ? tail_buffer_capacity : buffer_capacity);
      }
    });

  if (tail_array) {
    if (tail_buffer_capacity == 0)
      return std::unique_ptr<::MPI_Datatype, DatatypeDeleter>();
    buffer_capacity = tail_buffer_capacity;
  }

  auto buffer_indexer =
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, buffer);

  auto result = datatype(MPI_CXX_FLOAT_COMPLEX);
  for (auto ax = ms_shape.crbegin(); ax != ms_shape.crend(); ++ax) {
    auto ax_id = ax->id();
    auto ip =
      std::find_if(
        std::begin(*iter_params),
        std::end(*iter_params),
        [&ax_id](const IterParams& ip) {
          return ip.axis == ax_id;
        });
    if (ip->fully_in_array || ip->buffer_capacity > 0) {
      auto count = ip->fully_in_array ? ip->true_length() : buffer_capacity;
      if (count > 1) {
        auto i0 = buffer_indexer->offset_of_(index);
        ++index[ax_id];
        auto i1 = buffer_indexer->offset_of_(index);
        --index[ax_id];
        if (debug_log) {
          std::clog << "(" << rank << ") "
                    << mscol_nickname(ip->axis)
                    << " dv stride " << i1 - i0
                    << std::endl;
        }
        auto stride = (i1 - i0) * sizeof(std::complex<float>);
        auto dt = std::move(result);
        result = datatype();
        mpi_call(
          ::MPI_Type_create_hvector,
          count,
          1,
          stride,
          *dt,
          result.get());
      }
    }
  }
  mpi_call(::MPI_Type_commit, result.get());
  return result;
}

std::unique_ptr<::MPI_Datatype, DatatypeDeleter>
Reader::init_fileview(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::shared_ptr<std::vector<IterParams> >& iter_params,
  const std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
  bool tail_fileview,
  int rank,
  bool debug_log) {

  bool needs_tail = false;
  ArrayIndexer<MSColumns>::index index;
  std::size_t tail_size;
  std::for_each(
    std::begin(*iter_params),
    std::end(*iter_params),
    [&](const IterParams& ip) {
      index[ip.axis] = 0;
      if (ip.buffer_capacity > 0) {
        tail_size = ip.true_length() % ip.buffer_capacity;
        needs_tail = (tail_size != 0);
      }
    });

  if (tail_fileview && !needs_tail)
    return std::unique_ptr<::MPI_Datatype, DatatypeDeleter>();

  // build datatype for fileview
  auto result = datatype(MPI_CXX_FLOAT_COMPLEX);

  auto ms_axis = ms_shape.crbegin();
  while (ms_axis != ms_shape.crend()) {
    auto ms_axis_id = ms_axis->id();
    auto ip =
      std::find_if(
        std::begin(*iter_params),
        std::end(*iter_params),
        [&ms_axis_id](const IterParams& ip) {
          return ip.axis == ms_axis_id;
        });

    if (ip->within_fileview || ip->buffer_capacity > 1) {
      std::size_t count, num_blocks, block_len, terminal_block_len;
      if (ip->within_fileview) {
        count = ip->true_length();
        num_blocks = ip->max_blocks;
        block_len = ip->block_len;
        terminal_block_len = ip->terminal_block_len;
      } else {
        count =
          (tail_fileview
           ? ip->true_length() % ip->buffer_capacity
           : ip->buffer_capacity);
        num_blocks = (count + ip->block_len - 1) / ip->block_len;
        block_len = std::min(count, ip->block_len);
        terminal_block_len = (
          (count % block_len) > 0
          ? count % block_len
          : block_len);
      }
      if (count > 1) {
        // resize current fileview_datatype to equal stride between elements
        // on this axis
        auto i0 = ms_indexer->offset_of_(index);
        ++index[ip->axis];
        auto i1 = ms_indexer->offset_of_(index);
        --index[ip->axis];
        auto unit_stride = (i1 - i0) * sizeof(std::complex<float>);
        auto dt1 = std::move(result);
        ::MPI_Count size;
        mpi_call(::MPI_Type_size_x, *dt1, &size);
        ::MPI_Aint lb1, extent1;
        mpi_call(::MPI_Type_get_extent, *dt1, &lb1, &extent1);
        assert(static_cast<::MPI_Aint>(static_cast<std::size_t>(extent1))
               == extent1);
        if (static_cast<std::size_t>(extent1) != unit_stride) {
          result = datatype();
          mpi_call(
            ::MPI_Type_create_resized,
            *dt1,
            lb1,
            unit_stride,
            result.get());
          dt1 = std::move(result);
        }

        // create blocked vector of fileview_datatype elements
        index[ip->axis] += ip->stride;
        auto is = ms_indexer->offset_of_(index);
        index[ip->axis] -= ip->stride;
        auto stride = (is - i0) * sizeof(std::complex<float>);
        if (terminal_block_len == block_len) {
          // buffer capacity is assumed to be a multiple of block_len
          assert(ip->buffer_capacity % ip->block_len == 0);
          result = datatype();
          mpi_call(
            ::MPI_Type_create_hvector,
            num_blocks,
            block_len,
            stride,
            *dt1,
            result.get());
        } else {
          // first max_blocks - 1 blocks have one size, the last
          // block has a different size
          assert(num_blocks > 1);
          auto dt2 = datatype();
          mpi_call(
            ::MPI_Type_create_hvector,
            num_blocks - 1,
            block_len,
            stride,
            *dt1,
            dt2.get());
          if (terminal_block_len > 0) {
            auto dt3 = datatype();
            mpi_call(
              ::MPI_Type_contiguous,
              terminal_block_len,
              *dt1,
              dt3.get());
            auto terminal_displacement = stride * (num_blocks - 1);
            std::vector<int> blocklengths {1, 1};
            std::vector<::MPI_Aint> displacements {
              0, static_cast<::MPI_Aint>(terminal_displacement)};
            std::vector<MPI_Datatype> types {*dt2, *dt3};
            result = datatype();
            mpi_call(
              ::MPI_Type_create_struct,
              2,
              blocklengths.data(),
              displacements.data(),
              types.data(),
              result.get());
          } else {
            result = std::move(dt2);
          }
        }
      }
    }
    ++ms_axis;
  }

  if (debug_log) {
    ::MPI_Count size;
    mpi_call(::MPI_Type_size_x, *result, &size);
    std::clog << "(" << rank
              << ") fileview datatype size "
              << size / sizeof(std::complex<float>)
              << std::endl;
  }

  mpi_call(::MPI_Type_commit, result.get());
  return result;
}

void
Reader::set_fileview(::MPI_File file) const {
  // assume that m_mtx and m_mpi_state.handles() are locked
  std::size_t offset =
    m_ms_indexer->offset_of_(m_traversal_state.data_index);

  if (m_debug_log) {
    std::ostringstream oss;
    oss << "(" << m_rank << ") fv offset " << offset;
    auto index = m_traversal_state.data_index;
    std::for_each(
      std::begin(*m_iter_params),
      std::end(*m_iter_params),
      [&index, &oss](const IterParams& ip) {
        oss << "; " << mscol_nickname(ip.axis) << " " << index.at(ip.axis);
      });
    oss << std::endl;
    std::clog << oss.str();
  }
  const ::MPI_Datatype dt = *m_traversal_state.fileview_datatype();

  mpi_call(
    ::MPI_File_set_view,
    file,
    offset * sizeof(std::complex<float>),
    MPI_CXX_FLOAT_COMPLEX,
    dt,
    "native",
    MPI_INFO_NULL);
}

std::vector<IndexBlockSequenceMap<MSColumns> >
Reader::make_index_block_sequences(
  const std::shared_ptr<const std::vector<IterParams> >& iter_params) {
  std::vector<IndexBlockSequenceMap<MSColumns> > result;
  std::for_each(
    std::begin(*iter_params),
    std::end(*iter_params),
    [&result](const IterParams& ip) {
      std::vector<std::vector<IndexBlock> > blocks;
      if (ip.fully_in_array || ip.buffer_capacity > 0) {
        // merge gap-less consecutive blocks
        std::vector<IndexBlock> merged_blocks;
        std::size_t start = ip.origin;
        std::size_t end = ip.origin + ip.block_len;
        for (std::size_t b = 1; b < ip.max_blocks; ++b) {
          std::size_t s = ip.origin + b * ip.stride;
          if (s > end) {
            merged_blocks.emplace_back(start, end - start);
            start = s;
          }
          end = s + ip.block_len;
        }
        end = end - ip.block_len + ip.terminal_block_len;
        merged_blocks.emplace_back(start, end - start);

        // when entire axis doesn't fit into the array, we might have to split
        // blocks
        if (!ip.fully_in_array && ip.true_length() > ip.buffer_capacity) {
          std::size_t rem = ip.buffer_capacity;
          std::vector<IndexBlock> ibs;
          std::for_each(
            std::begin(merged_blocks),
            std::end(merged_blocks),
            [&ip, &blocks, &rem, &ibs](IndexBlock& blk) {
              while (blk.m_length > 0) {
                std::size_t len = std::min(rem, blk.m_length);
                ibs.emplace_back(blk.m_index, len);
                blk = IndexBlock(blk.m_index + len, blk.m_length - len);
                rem -= len;
                if (rem == 0) {
                  blocks.push_back(ibs);
                  ibs.clear();
                  rem = ip.buffer_capacity;
                }
              }
            });
          if (ibs.size() > 0)
            blocks.push_back(ibs);
        } else {
          blocks.push_back(merged_blocks);
        }
      } else {
        blocks.push_back(std::vector<IndexBlock>{ IndexBlock{ ip.origin, 1 } });
      }
      result.emplace_back(ip.axis, blocks);
    });
  return result;
}

std::unique_ptr<std::complex<float> >
Reader::read_arrays(
  const std::vector<IndexBlockSequence<MSColumns> >& blocks,
  ::MPI_File file) const {
  // assume that m_mtx and m_mpi_state.handles() are locked

  if (m_debug_log) {
    std::clog << "(" << m_rank << ") read ";
    const char* sep0 = "";
    std::for_each(
      std::begin(blocks),
      std::end(blocks),
      [&sep0](auto& ibs) {
        std::clog << sep0
                  << mscol_nickname(ibs.m_axis) << ": [";
        sep0 = "; ";
        const char *sep1 = "";
        std::for_each(
          std::begin(ibs.m_blocks),
          std::end(ibs.m_blocks),
          [&sep1](auto& b) {
            std::clog << sep1 << "(" << b.m_index << "," << b.m_length << ")";
            sep1 = ",";
          });
        std::clog << "]";
      });
    std::clog << std::endl;
  }

  int count;
  std::unique_ptr<std::complex<float> > result;
  if (m_traversal_state.count > 0) {
    result.reset(
      reinterpret_cast<std::complex<float> *>(::operator new(m_buffer_size)));
    count = 1;
  } else {
    count = 0;
  }

  auto dt = *m_traversal_state.array_datatype();

  ::MPI_Status status;
  mpi_call(
    ::MPI_File_read_all,
    file,
    result.get(),
    count,
    dt,
    &status);

  int st_count;
  mpi_call(::MPI_Get_count, &status, dt, &st_count);
  if (count != st_count) {
    result.reset();
    if (m_debug_log)
      std::clog << "(" << m_rank << ") "
                << "expected " << count
                << ", got " << st_count
                << std::endl;
  }
  return result;
}

void
Reader::advance_to_next_buffer(::MPI_File file) {
  // assume that m_mtx and m_mpi_state.handles() are locked
  m_traversal_state.count = 0;
  m_traversal_state.max_count = 0;
  while (!m_traversal_state.eof && !m_traversal_state.axis_iters.empty()) {
    AxisIter& axis_iter = m_traversal_state.axis_iters.top();
    MSColumns axis = axis_iter.params->axis;
    if (!axis_iter.at_end) {
      auto depth = m_traversal_state.axis_iters.size();
      m_traversal_state.data_index[axis] = axis_iter.index;
      if (axis_iter.params->buffer_capacity > 0) {
        m_traversal_state.max_count =
          static_cast<int>(axis_iter.params->buffer_capacity);
        m_traversal_state.count =
          static_cast<int>(
            std::min(
              axis_iter.num_remaining(),
              axis_iter.params->buffer_capacity));
        m_traversal_state.in_tail =
          m_traversal_state.count < m_traversal_state.max_count;
      } else {
        m_traversal_state.in_tail = false;
      }
      if (*m_inner_fileview_axis && axis == m_inner_fileview_axis->value())
        set_fileview(file);
      if (axis_iter.params->buffer_capacity > 0)
        return;
      const IterParams* next_params = &(*m_iter_params)[depth];
      m_traversal_state.axis_iters.emplace(
        std::shared_ptr<const IterParams>(m_iter_params, next_params),
        axis_iter.at_data);
    } else {
      m_traversal_state.data_index[axis] = axis_iter.params->origin;
      m_traversal_state.axis_iters.pop();
      if (!m_traversal_state.axis_iters.empty())
        m_traversal_state.axis_iters.top().increment();
    }
  }
  return;
}

void
Reader::advance_to_buffer_end() {
  // assume that m_mtx is locked
  AxisIter* axis_iter = &m_traversal_state.axis_iters.top();
  axis_iter->increment(m_traversal_state.max_count);
  m_traversal_state.data_index[axis_iter->params->axis] = axis_iter->index;
  while (
    !m_traversal_state.eof
    && !m_traversal_state.axis_iters.empty()
    && axis_iter->at_end) {

    m_traversal_state.data_index[axis_iter->params->axis] =
      axis_iter->params->origin;
    m_traversal_state.axis_iters.pop();
    if (!m_traversal_state.axis_iters.empty()) {
      axis_iter = &m_traversal_state.axis_iters.top();
      axis_iter->increment();
      m_traversal_state.data_index[axis_iter->params->axis] = axis_iter->index;
    }
  }
}

void
Reader::read_next_buffer(::MPI_File file) {

  advance_to_next_buffer(file);
  std::shared_ptr<std::complex<float> > buffer;
  auto blocks = m_traversal_state.blocks();
  buffer = read_arrays(blocks, file);
  m_ms_array = MSArray(std::move(blocks), buffer);
}
