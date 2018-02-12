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

Reader::Reader()
  : m_readahead(false) {
  m_ms_array = std::make_tuple(
    MSArray(),
    std::variant<::MPI_Request, ::MPI_Status>(std::in_place_index<1>),
    0);

  m_next_ms_array = std::make_tuple(
    MSArray(),
    std::variant<::MPI_Request, ::MPI_Status>(std::in_place_index<1>),
    0);
}

Reader::Reader(
  MPIState&& mpi_state,
  std::shared_ptr<const std::vector<ColumnAxisBase<MSColumns> > > ms_shape,
  std::shared_ptr<const std::vector<IterParams> > iter_params,
  std::shared_ptr<const std::vector<MSColumns> > buffer_order,
  std::shared_ptr<const std::optional<MSColumns> > inner_fileview_axis,
  std::shared_ptr<const ArrayIndexer<MSColumns> > ms_indexer,
  std::size_t buffer_size,
  bool readahead,
  TraversalState&& traversal_state,
  bool debug_log)
  : m_mpi_state(std::move(mpi_state))
  , m_ms_shape(ms_shape)
  , m_iter_params(iter_params)
  , m_buffer_order(buffer_order)
  , m_inner_fileview_axis(inner_fileview_axis)
  , m_etype_datatype(datatype(MPI_CXX_FLOAT_COMPLEX))
  , m_ms_indexer(ms_indexer)
  , m_buffer_size(buffer_size)
  , m_readahead(readahead)
  , m_debug_log(debug_log)
  , m_traversal_state(std::move(traversal_state)) {

  auto handles = m_mpi_state.handles();
  mpi_call(::MPI_Comm_rank, handles->comm, reinterpret_cast<int*>(&m_rank));

  m_ms_array = std::make_tuple(
    MSArray(),
    std::variant<::MPI_Request, ::MPI_Status>(std::in_place_index<1>),
    0);

  m_next_ms_array = std::make_tuple(
    MSArray(),
    std::variant<::MPI_Request, ::MPI_Status>(std::in_place_index<1>),
    0);

  if (handles->file != MPI_FILE_NULL) {
    if (!*m_inner_fileview_axis)
      set_fileview(m_traversal_state, handles->file);
    const IterParams* init_params = &(*m_iter_params)[0];
    m_traversal_state.axis_iters.emplace(
      std::shared_ptr<const IterParams>(m_iter_params, init_params),
      init_params->max_blocks > 0);
    m_ms_array =
      read_next_buffer(m_traversal_state, m_readahead, handles->file);
    if (m_readahead)
      start_next();
  } else {
    m_traversal_state.eof = true;
  }

}

Reader::~Reader() {
  if (m_readahead) {
    wait_for_array(m_ms_array, true);
    wait_for_array(m_next_ms_array, true);
  }
}

Reader
Reader::begin(
  const std::string& path,
  ::MPI_Comm comm,
  ::MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  bool ms_buffer_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t max_buffer_size,
  bool readahead,
  bool debug_log) {

  ::MPI_Comm reduced_comm;
  ::MPI_File file;
  ::MPI_Info priv_info;
  std::shared_ptr<std::vector<IterParams> > iter_params;
  std::shared_ptr<std::vector<MSColumns> > buffer_order;
  std::shared_ptr<std::optional<MSColumns> > inner_fileview_axis;
  std::shared_ptr<ArrayIndexer<MSColumns> > ms_indexer;
  std::size_t buffer_size;
  TraversalState traversal_state;

  initialize(
    path,
    comm,
    info,
    ms_shape,
    traversal_order,
    ms_buffer_order,
    pgrid,
    max_buffer_size,
    debug_log,
    MPI_MODE_RDONLY,
    reduced_comm,
    priv_info,
    file,
    iter_params,
    buffer_order,
    inner_fileview_axis,
    ms_indexer,
    buffer_size,
    traversal_state);

  return
    Reader(
      MPIState(reduced_comm, priv_info, file, path),
      std::make_shared<std::vector<ColumnAxisBase<MSColumns> > >(ms_shape),
      iter_params,
      buffer_order,
      inner_fileview_axis,
      ms_indexer,
      buffer_size,
      readahead,
      std::move(traversal_state),
      debug_log);
}

Reader
Reader::wbegin(
  const std::string& path,
  ::MPI_Comm comm,
  ::MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  bool ms_buffer_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t max_buffer_size,
  bool readahead,
  bool debug_log) {

  ::MPI_Comm reduced_comm;
  ::MPI_File file;
  ::MPI_Info priv_info;
  std::shared_ptr<std::vector<IterParams> > iter_params;
  std::shared_ptr<std::vector<MSColumns> > buffer_order;
  std::shared_ptr<std::optional<MSColumns> > inner_fileview_axis;
  std::shared_ptr<ArrayIndexer<MSColumns> > ms_indexer;
  std::size_t buffer_size;
  TraversalState traversal_state;

  initialize(
    path,
    comm,
    info,
    ms_shape,
    traversal_order,
    ms_buffer_order,
    pgrid,
    max_buffer_size,
    debug_log,
    MPI_MODE_CREATE | MPI_MODE_RDWR,
    reduced_comm,
    priv_info,
    file,
    iter_params,
    buffer_order,
    inner_fileview_axis,
    ms_indexer,
    buffer_size,
    traversal_state);

  std::size_t total_length = 1;
  std::for_each(
    std::begin(ms_shape),
    std::end(ms_shape),
    [&total_length](auto& ax) {
      total_length *= ax.length();
    });
  mpi_call(
    ::MPI_File_set_size,
    file,
    sizeof(std::complex<float>) * total_length);

  return
    Reader(
      MPIState(reduced_comm, priv_info, file, path),
      std::make_shared<std::vector<ColumnAxisBase<MSColumns> > >(ms_shape),
      iter_params,
      buffer_order,
      inner_fileview_axis,
      ms_indexer,
      buffer_size,
      readahead,
      std::move(traversal_state),
      debug_log);
}

void
Reader::initialize(
  const std::string& path,
  ::MPI_Comm comm,
  ::MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  bool ms_buffer_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t max_buffer_size,
  bool debug_log,
  int amode,
  ::MPI_Comm& reduced_comm,
  ::MPI_Info& priv_info,
  ::MPI_File& file,
  std::shared_ptr<std::vector<IterParams> >& iter_params,
  std::shared_ptr<std::vector<MSColumns> >& buffer_order,
  std::shared_ptr<std::optional<MSColumns> >& inner_fileview_axis,
  std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
  std::size_t& buffer_size,
  TraversalState& traversal_state) {

  buffer_size =
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

  iter_params =
    std::make_shared<std::vector<IterParams> >(traversal_order.size());
  inner_fileview_axis = std::make_shared<std::optional<MSColumns> >();
  ms_indexer = ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, ms_shape);
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
    std::ostringstream out;
    out << "(" << rank << ") "
        << "outer full array axis "
        << mscol_nickname(outer_full_array_axis)
        << std::endl;
    out << "(" << rank << ") "
        << "read " << array_read->buffer_capacity
        << " arrays at " << mscol_nickname(array_read->axis)
        << std::endl;
    if (inner_fileview_axis->has_value())
      out << "(" << rank << ") "
          << "m_inner_fileview_axis "
          << mscol_nickname(inner_fileview_axis->value())
          << std::endl;
    else
      out << "(" << rank << ") "
          << "no inner_fileview_axis"
          << std::endl;
    std::clog << out.str();
  }

  std::shared_ptr<::MPI_Datatype> full_buffer_datatype;
  unsigned full_buffer_dt_count;
  std::tie(full_buffer_datatype, full_buffer_dt_count) =
    init_buffer_datatype(
      ms_shape,
      iter_params,
      ms_buffer_order,
      false,
      rank,
      debug_log);

  std::shared_ptr<::MPI_Datatype> tail_buffer_datatype;
  unsigned tail_buffer_dt_count;
  std::tie(tail_buffer_datatype, tail_buffer_dt_count) =
    init_buffer_datatype(
      ms_shape,
      iter_params,
      ms_buffer_order,
      true,
      rank,
      debug_log);
  if (!tail_buffer_datatype) {
    tail_buffer_datatype = full_buffer_datatype;
    tail_buffer_dt_count = full_buffer_dt_count;
  }

  buffer_order =
    std::shared_ptr<std::vector<MSColumns> >(new std::vector<MSColumns>);
  if (ms_buffer_order)
    std::for_each(
      std::begin(ms_shape),
      std::end(ms_shape),
      [&iter_params, &buffer_order](auto& ax) {
        auto ip = find_iter_params(iter_params, ax.id());
        if (ip != nullptr && (ip->buffer_capacity > 0 || ip->fully_in_array))
          buffer_order->emplace_back(ip->axis);
      });
  else
    std::for_each(
      std::begin(*iter_params),
      std::end(*iter_params),
      [&buffer_order](auto &ip) {
        if (ip.buffer_capacity > 0 || ip.fully_in_array)
          buffer_order->emplace_back(ip.axis);
      });

  file = MPI_FILE_NULL;
  priv_info = info;
  if (info != MPI_INFO_NULL)
    mpi_call(::MPI_Info_dup, info, &priv_info);
  if (reduced_comm != MPI_COMM_NULL) {
    mpi_call(
      ::MPI_File_open,
      reduced_comm,
      path.c_str(),
      amode,
      priv_info,
      &file);
    mpi_call(::MPI_File_set_errhandler, file, MPI_ERRORS_RETURN);
    if (debug_log && rank == 0) {
      MPI_Info info_used;
      mpi_call(::MPI_File_get_info, file, &info_used);
      std::array<char,80> value;
      int flag;
      mpi_call(
        ::MPI_Info_get,
        info_used,
        "romio_filesystem_type",
        value.size() - 1,
        value.data(),
        &flag);
      if (flag != 0)
        std::clog << "ROMIO filesystem type: " << value.data() << std::endl;
      else
        std::clog << "No ROMIO filesystem type" << std::endl;
      mpi_call(::MPI_Info_free, &info_used);
    }
  }

  std::shared_ptr<::MPI_Datatype> full_fileview_datatype =
    init_fileview(
      file,
      ms_shape,
      iter_params,
      ms_indexer,
      false,
      rank,
      debug_log);

  std::shared_ptr<::MPI_Datatype> tail_fileview_datatype =
    init_fileview(
      file,
      ms_shape,
      iter_params,
      ms_indexer,
      true,
      rank,
      debug_log);
  if (!tail_fileview_datatype)
    tail_fileview_datatype = full_fileview_datatype;

  traversal_state =
    TraversalState(
      reduced_comm,
      iter_params,
      outer_full_array_axis,
      full_buffer_datatype,
      full_buffer_dt_count,
      tail_buffer_datatype,
      tail_buffer_dt_count,
      full_fileview_datatype,
      tail_fileview_datatype);

  std::for_each(
    std::begin(*iter_params),
    std::end(*iter_params),
    [&traversal_state](const IterParams& ip) {
      traversal_state.data_index[ip.axis] = ip.origin;
    });
}

void
Reader::step(bool cont) {
  std::lock_guard<decltype(m_mtx)> lock(m_mtx);

  if (at_end())
    throw std::out_of_range("Reader cannot be advanced: at end");

  if (m_readahead) {
    wait_for_array(m_ms_array, !cont);
    m_traversal_state = std::move(m_next_traversal_state);
    std::swap(m_ms_array, m_next_ms_array);
    if (at_end())
      return;
  }

  m_traversal_state.cont = cont;
  start_next();

  if (!m_readahead) {
    m_traversal_state = std::move(m_next_traversal_state);
    std::swap(m_ms_array, m_next_ms_array);
  }
}

void
Reader::start_next() {
  m_next_traversal_state = m_traversal_state;
  advance_to_buffer_end(m_next_traversal_state);

  m_next_traversal_state.eof = m_next_traversal_state.axis_iters.empty();
  std::array<bool, 2>
    tests{ m_next_traversal_state.cont, m_next_traversal_state.eof };
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
  m_next_traversal_state.cont = tests[0];
  m_next_traversal_state.eof = tests[1];
  if (m_next_traversal_state.cont && !m_next_traversal_state.eof) {
    // wait for previous I/O request to complete...I don't know that this
    // requirement isn't a bug in a romio/mpich/mvapich non-blocking function
    // implementation...without waiting, "reader-test" using a single process
    // with readahead fails in such a way that it appears that the same buffers
    // are returned for some consecutive reads, but there is no failure when the
    // number of ranks is "high enough"
    wait_for_array(m_ms_array);
    m_next_ms_array =
      read_next_buffer(m_next_traversal_state, m_readahead, handles->file);
  } else {
    m_next_ms_array =
      std::make_tuple(
        MSArray(),
        std::variant<::MPI_Request,::MPI_Status>(std::in_place_index<1>),
        0);
  }
}

bool
Reader::at_end() const {
  return !m_traversal_state.cont
    || m_traversal_state.eof
    || m_traversal_state.axis_iters.empty();
}

std::size_t
Reader::buffer_length() const {
  std::lock_guard<decltype(m_mtx)> lock(m_mtx);
  auto indices = this->indices();
  std::size_t result = 1;
  std::for_each(
    std::begin(indices),
    std::end(indices),
    [&result](auto& ibs) {
      result *= ibs.num_elements();
    });
  return result;
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
  swap(m_readahead, other.m_readahead);
  swap(m_iter_params, other.m_iter_params);
  swap(m_buffer_order, other.m_buffer_order);
  swap(m_etype_datatype, other.m_etype_datatype);
  swap(m_inner_fileview_axis, other.m_inner_fileview_axis);
  swap(m_ms_indexer, other.m_ms_indexer);
  swap(m_debug_log, other.m_debug_log);
  swap(m_traversal_state, other.m_traversal_state);
  swap(m_next_traversal_state, other.m_next_traversal_state);
  swap(m_ms_array, other.m_ms_array);
  swap(m_next_ms_array, other.m_next_ms_array);
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
    std::ostringstream out;
    std::for_each(
      std::begin(*iter_params),
      std::end(*iter_params),
      [&out, &rank](auto& ip) {
        out << "(" << rank << ") "
            << mscol_nickname(ip.axis)
            << " capacity " << ip.buffer_capacity
            << ", fully_in " << ip.fully_in_array
            << ", array_len " << ip.array_length
            << std::endl;
      });
    std::clog << out.str();
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
    bool ooo = false;
    auto ip = iter_params->rbegin();
    while (ip != iter_params->rend()) {
      if (!*inner_fileview_axis) {
        ooo = ooo || out_of_order.count(ip->axis) > 0;
        if (ooo
            && !(ip->fully_in_array
                 || ip->buffer_capacity == ip->block_len * ip->max_blocks))
          *inner_fileview_axis = ip->axis;
      }
      ip->within_fileview = !inner_fileview_axis->has_value();
      ++ip;
    }
  }
}

std::tuple<std::unique_ptr<::MPI_Datatype, DatatypeDeleter>, unsigned>
Reader::init_buffer_datatype(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::shared_ptr<std::vector<IterParams> >& iter_params,
  bool ms_buffer_order,
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

  // order iter_params to establish the axis ordering in buffer
  std::vector<const IterParams*> reordered_ips;
  if (ms_buffer_order)
    std::transform(
      std::begin(ms_shape),
      std::end(ms_shape),
      std::back_inserter(reordered_ips),
      [&iter_params](auto& ax) {
        return find_iter_params(iter_params, ax.id());
      });
  else
    std::transform(
      std::begin(*iter_params),
      std::end(*iter_params),
      std::back_inserter(reordered_ips),
      [](auto& ip) {
        return &ip;
      });

  // compute buffer shape
  std::size_t buffer_capacity = 0;
  std::size_t tail_buffer_capacity = 0;
  std::vector<ColumnAxisBase<MSColumns> > buffer;
  std::for_each(
    std::begin(reordered_ips),
    std::end(reordered_ips),
    [&](auto& ip) {
      if (ip->fully_in_array) {
        buffer.emplace_back(
          static_cast<unsigned>(ip->axis),
          ip->true_length());
      } else if (ip->buffer_capacity > 0) {
        buffer_capacity = ip->buffer_capacity;
        tail_buffer_capacity = ip->true_length() % ip->buffer_capacity;
        std::size_t capacity =
          tail_array ? tail_buffer_capacity : buffer_capacity;
        buffer.emplace_back(static_cast<unsigned>(ip->axis), capacity);
      }
    });

  if (tail_array) {
    if (tail_buffer_capacity == 0)
      return std::make_tuple(
        std::unique_ptr<::MPI_Datatype, DatatypeDeleter>(),
        0);
    buffer_capacity = tail_buffer_capacity;
  }

  auto buffer_indexer =
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, buffer);

  auto result_dt = datatype(MPI_CXX_FLOAT_COMPLEX);
  unsigned result_dt_count = 1;
  std::for_each(
    ms_shape.crbegin(),
    ms_shape.crend(),
    [&](auto& ax) {
      auto ip = find_iter_params(iter_params, ax.id());
      if (ip->fully_in_array || ip->buffer_capacity > 1) {
        auto count = ip->fully_in_array ? ip->true_length() : buffer_capacity;
        auto i0 = buffer_indexer->offset_of_(index);
        ++index[ip->axis];
        auto i1 = buffer_indexer->offset_of_(index);
        --index[ip->axis];
        if (debug_log) {
          std::clog << "(" << rank << ") "
                    << mscol_nickname(ip->axis)
                    << " dv stride " << i1 - i0
                    << std::endl;
        }
        auto stride = (i1 - i0) * sizeof(std::complex<float>);
        ::MPI_Aint lb = 0, extent = 0;
        mpi_call(::MPI_Type_get_extent, *result_dt, &lb, &extent);
        if (stride == result_dt_count * static_cast<std::size_t>(extent)) {
          result_dt_count *= count;
        } else {
          auto dt = std::move(result_dt);
          result_dt = datatype();
          mpi_call(
            ::MPI_Type_create_hvector,
            count,
            result_dt_count,
            stride,
            *dt,
            result_dt.get());
          result_dt_count = 1;
        }
      }
    });

  if (debug_log) {
    ::MPI_Count size;
    mpi_call(::MPI_Type_size_x, *result_dt, &size);
    std::clog << "(" << rank << ") buffer datatype size "
              << size / sizeof(std::complex<float>)
              << ", count " << result_dt_count << std::endl;
  }
  if (*result_dt != MPI_CXX_FLOAT_COMPLEX)
    mpi_call(::MPI_Type_commit, result_dt.get());
  return std::make_tuple(std::move(result_dt), result_dt_count);
}

std::unique_ptr<::MPI_Datatype, DatatypeDeleter>
Reader::init_fileview(
  ::MPI_File file,
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
    ms_shape.crbegin(),
    ms_shape.crend(),
    [&](auto& ax) {
      auto ip = find_iter_params(iter_params, ax.id());
      index[ip->axis] = 0;
      if (!ip->within_fileview && ip->buffer_capacity > 0) {
        tail_size = ip->true_length() % ip->buffer_capacity;
        needs_tail = (tail_size != 0);
      }
    });

  if (tail_fileview && !needs_tail)
    return std::unique_ptr<::MPI_Datatype, DatatypeDeleter>();

  // build datatype for fileview
  auto result = datatype(MPI_CXX_FLOAT_COMPLEX);
  auto committed = true;
  unsigned dt_count = 1;
  ::MPI_Aint value_extent;
  mpi_call(::MPI_File_get_type_extent, file, *result, &value_extent);

  bool prev_within_fileview = true;
  auto ms_axis = ms_shape.crbegin();
  do {
    auto ip = (
      (ms_axis != ms_shape.crend())
      ? find_iter_params(iter_params, ms_axis->id())
      : nullptr);
    if ((ip && (ip->within_fileview || ip->buffer_capacity > 0))
        || (!ip && prev_within_fileview)) {
      std::size_t count, num_blocks, block_len, terminal_block_len, unit_stride;
      if (ip) {
        if (ip->within_fileview) {
          count = ip->true_length();
          num_blocks = ip->max_blocks;
          block_len = ip->block_len;
          terminal_block_len = ip->terminal_block_len;
          prev_within_fileview = true;
        } else if (ip->buffer_capacity > 0) {
          assert(ip->buffer_capacity % ip->block_len == 0);
          num_blocks = ip->buffer_capacity / ip->block_len;
          block_len = ip->block_len;
          if (!tail_fileview) {
            count = ip->buffer_capacity;
            terminal_block_len = block_len;
          } else {
            count = tail_size;
            terminal_block_len = tail_size % ip->block_len;
          }
          prev_within_fileview = ip->true_length() <= ip->buffer_capacity;
        } else {
          assert(false);
        }
        auto i0 = ms_indexer->offset_of_(index);
        ++index[ip->axis];
        auto i1 = ms_indexer->offset_of_(index);
        --index[ip->axis];
        unit_stride = i1 - i0;
      } else /* prev_within_fileview */ {
        count = 1;
        num_blocks = 1;
        block_len = 1;
        terminal_block_len = 1;
        unit_stride = 1;
        std::for_each(
          std::begin(ms_shape),
          std::end(ms_shape),
          [&unit_stride](auto& ax) { unit_stride *= ax.length(); });
        prev_within_fileview = false;
      }

      // resize current fileview_datatype to equal stride between elements
      // on this axis
      auto unit_hstride = unit_stride * value_extent;
      auto dt1 = std::move(result);
      if (!committed)
        mpi_call(::MPI_Type_commit, dt1.get());
      committed = true;
      ::MPI_Aint extent;
      mpi_call(::MPI_File_get_type_extent, file, *dt1, &extent);
      if (dt_count * static_cast<std::size_t>(extent) != unit_hstride) {
        if (dt_count > 1) {
          result = datatype();
          mpi_call(::MPI_Type_contiguous, dt_count, *dt1, result.get());
          dt1 = std::move(result);
          dt_count = 1;
        }
        result = datatype();
        committed = false;
        mpi_call(
          ::MPI_Type_create_resized,
          *dt1,
          0,
          unit_hstride,
          result.get());
        dt1 = std::move(result);
      }

      if (count == 1) {
        result = std::move(dt1);
      } else {
        // buffer capacity is assumed to be a multiple of block_len
        assert(ip->buffer_capacity % ip->block_len == 0);

        // create (blocked) vector of fileview_datatype elements
        auto i0 = ms_indexer->offset_of_(index);
        index[ip->axis] += ip->stride;
        auto is = ms_indexer->offset_of_(index);
        index[ip->axis] -= ip->stride;
        auto stride = (is - i0) / unit_stride;
        if (block_len == stride) {
          dt_count *= count;
          result = std::move(dt1);
        } else if (terminal_block_len == block_len) {
          // uniform blocked vector
          result = datatype();
          committed = false;
          mpi_call(
            ::MPI_Type_vector,
            num_blocks,
            block_len * dt_count,
            stride * dt_count,
            *dt1,
            result.get());
          dt_count = 1;
        } else {
          assert(num_blocks > 1);
          // avoid using MPI_Type_create_struct to maintain a portable datatype
          result = datatype();
          committed = false;
          if (terminal_block_len == 0) {
            mpi_call(
              ::MPI_Type_vector,
              num_blocks - 1,
              block_len * dt_count,
              stride * dt_count,
              *dt1,
              result.get());
          } else {
#ifndef USE_INDEXED
            // first max_blocks - 1 blocks have one size, the last
            // block has a different size
            assert(num_blocks > 1);
            auto dt2 = datatype();
            mpi_call(
              ::MPI_Type_vector,
              num_blocks - 1,
              block_len * dt_count,
              stride * dt_count,
              *dt1,
              dt2.get());
            if (terminal_block_len > 0) {
              auto dt3 = datatype();
              mpi_call(
                ::MPI_Type_contiguous,
                terminal_block_len * dt_count,
                *dt1,
                dt3.get());
              auto hstride = (is - i0) * value_extent;
              auto terminal_displacement = hstride * (num_blocks - 1);
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
#else // USE_INDEXED
            auto blocklengths = std::make_unique<int[]>(num_blocks);
            auto displacements = std::make_unique<int[]>(num_blocks);
            for (std::size_t i = 0; i < num_blocks; ++i) {
              blocklengths[i] = block_len * dt_count;
              displacements[i] = i * stride * dt_count;
            }
            blocklengths[num_blocks - 1] = terminal_block_len * dt_count;
            mpi_call(
              ::MPI_Type_indexed,
              num_blocks,
              blocklengths.get(),
              displacements.get(),
              *dt1,
              result.get());
#endif
          }
          dt_count = 1;
        }
      }
    }
    ++ms_axis;
  } while (ms_axis != ms_shape.crend());

  if (dt_count > 1) {
    auto dt1 = std::move(result);
    result = datatype();
    committed = false;
    mpi_call(::MPI_Type_contiguous, dt_count, *dt1, result.get());
    dt_count = 1;
  }

  if (!committed)
    mpi_call(::MPI_Type_commit, result.get());

  if (debug_log) {
    ::MPI_Count size;
    mpi_call(::MPI_Type_size_x, *result, &size);
    std::clog << "(" << rank << ") fileview datatype size "
              << size / value_extent
              << ", count " << dt_count
              << std::endl;
  }
  return result;
}

void
Reader::set_fileview(TraversalState& traversal_state, ::MPI_File file) const {
  // assume that m_mtx and m_mpi_state.handles() are locked
  std::size_t offset =
    m_ms_indexer->offset_of_(traversal_state.data_index);

  if (m_debug_log) {
    std::ostringstream oss;
    oss << "(" << m_rank << ") fv offset " << offset;
    auto index = traversal_state.data_index;
    std::for_each(
      std::begin(*m_iter_params),
      std::end(*m_iter_params),
      [&index, &oss](const IterParams& ip) {
        oss << "; " << mscol_nickname(ip.axis) << " " << index.at(ip.axis);
      });
    oss << std::endl;
    std::clog << oss.str();
  }

  const ::MPI_Datatype dt = *traversal_state.fileview_datatype();

  mpi_call(
    ::MPI_File_set_view,
    file,
    offset * sizeof(std::complex<float>),
    MPI_CXX_FLOAT_COMPLEX,
    dt,
    "native",
    MPI_INFO_NULL);
}

std::unique_ptr<std::vector<IndexBlockSequenceMap<MSColumns> > >
Reader::make_index_block_sequences(
  const std::shared_ptr<const std::vector<IterParams> >& iter_params) {
  std::unique_ptr<std::vector<IndexBlockSequenceMap<MSColumns> > > result(
    new std::vector<IndexBlockSequenceMap<MSColumns> >());
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
      result->emplace_back(ip.axis, blocks);
    });
  return result;
}

std::tuple<
  std::unique_ptr<std::complex<float> >,
  std::variant<::MPI_Request, ::MPI_Status>,
  ::MPI_Offset>
Reader::read_arrays(
  TraversalState& traversal_state,
  bool nonblocking,
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

  std::shared_ptr<const ::MPI_Datatype> dt;
  unsigned count;
  std::tie(dt, count) = traversal_state.buffer_datatype();

  std::unique_ptr<std::complex<float> > buffer;
  if (traversal_state.count > 0)
    buffer.reset(
      reinterpret_cast<std::complex<float> *>(::operator new(m_buffer_size)));
  else
    count = 0;

  ::MPI_Offset start;
  mpi_call(::MPI_File_get_position, file, &start);

  if (!nonblocking) {
    std::tuple<
      std::unique_ptr<std::complex<float> >,
      std::variant<::MPI_Request, ::MPI_Status>,
      ::MPI_Offset> result(
        std::move(buffer),
        std::variant<::MPI_Request, ::MPI_Status>(std::in_place_index<1>),
        start);
    mpi_call(
      ::MPI_File_read_all,
      file,
      std::get<0>(result).get(),
      count,
      *dt,
      &std::get<::MPI_Status>(std::get<1>(result)));

    // TODO: move this block
    int st_count;
    mpi_call(
      ::MPI_Get_count,
      &std::get<::MPI_Status>(std::get<1>(result)),
      *dt,
      &st_count);
    if (count != static_cast<unsigned>(st_count)) {
      buffer.reset();
      if (m_debug_log)
        std::clog << "(" << m_rank << ") "
                  << "expected " << count
                  << ", got " << st_count
                  << std::endl;
    }
    return result;
  } else {
    std::tuple<
      std::unique_ptr<std::complex<float> >,
      std::variant<::MPI_Request, ::MPI_Status>,
      ::MPI_Offset> result(
        std::move(buffer),
        std::variant<::MPI_Request, ::MPI_Status>(std::in_place_index<0>),
        start);
    mpi_call(
      ::MPI_File_iread_all,
      file,
      std::get<0>(result).get(),
      count,
      *dt,
      &std::get<::MPI_Request>(std::get<1>(result)));
    return result;
  }
}

void
Reader::advance_to_next_buffer(
  TraversalState& traversal_state,
  ::MPI_File file) const {
  // assume that m_mtx and m_mpi_state.handles() are locked
  traversal_state.count = 0;
  traversal_state.max_count = 0;
  while (!traversal_state.eof && !traversal_state.axis_iters.empty()) {
    AxisIter& axis_iter = traversal_state.axis_iters.top();
    MSColumns axis = axis_iter.params->axis;
    if (!axis_iter.at_end) {
      auto depth = traversal_state.axis_iters.size();
      traversal_state.data_index[axis] = axis_iter.index;
      if (axis_iter.params->buffer_capacity > 0) {
        traversal_state.max_count =
          static_cast<int>(axis_iter.params->buffer_capacity);
        traversal_state.count =
          static_cast<int>(
            std::min(
              axis_iter.num_remaining(),
              axis_iter.params->buffer_capacity));
        traversal_state.in_tail =
          traversal_state.count < traversal_state.max_count;
      } else {
        traversal_state.in_tail = false;
      }
      if (*m_inner_fileview_axis && axis == m_inner_fileview_axis->value()) {
        wait_for_array(m_ms_array);
        wait_for_array(m_next_ms_array);
        set_fileview(traversal_state, file);
      }
      if (axis_iter.params->buffer_capacity > 0)
        return;
      const IterParams* next_params = &(*m_iter_params)[depth];
      traversal_state.axis_iters.emplace(
        std::shared_ptr<const IterParams>(m_iter_params, next_params),
        axis_iter.at_data);
    } else {
      traversal_state.data_index[axis] = axis_iter.params->origin;
      traversal_state.axis_iters.pop();
      if (!traversal_state.axis_iters.empty())
        traversal_state.axis_iters.top().increment();
    }
  }
  return;
}

void
Reader::advance_to_buffer_end(TraversalState& traversal_state) const {
  // assume that m_mtx is locked
  AxisIter* axis_iter = &traversal_state.axis_iters.top();
  axis_iter->increment(traversal_state.max_count);
  traversal_state.data_index[axis_iter->params->axis] = axis_iter->index;
  while (
    !traversal_state.eof
    && !traversal_state.axis_iters.empty()
    && axis_iter->at_end) {

    traversal_state.data_index[axis_iter->params->axis] =
      axis_iter->params->origin;
    traversal_state.axis_iters.pop();
    if (!traversal_state.axis_iters.empty()) {
      axis_iter = &traversal_state.axis_iters.top();
      axis_iter->increment();
      traversal_state.data_index[axis_iter->params->axis] = axis_iter->index;
    }
  }
}

std::tuple<MSArray, std::variant<::MPI_Request, ::MPI_Status>, ::MPI_Offset>
Reader::read_next_buffer(
  TraversalState& traversal_state,
  bool nonblocking,
  ::MPI_File file) const {

  // assume that m_mtx is locked
  advance_to_next_buffer(traversal_state, file);
  auto blocks = traversal_state.blocks();
  std::stable_sort(
    std::begin(blocks),
    std::end(blocks),
    [this](const IndexBlockSequence<MSColumns>& ibs0,
           const IndexBlockSequence<MSColumns>& ibs1) {
      return buffer_order_compare(ibs0.m_axis, ibs1.m_axis);
    });
  std::shared_ptr<std::complex<float> > buffer;
  std::variant<::MPI_Request, ::MPI_Status> rs;
  ::MPI_Offset start;
  std::tie(buffer, rs, start) =
    read_arrays(traversal_state, nonblocking, blocks, file);
  return std::make_tuple(MSArray(std::move(blocks), buffer), rs, start);
}

bool
Reader::buffer_order_compare(const MSColumns& col0, const MSColumns& col1)
  const {

  auto p0 =
    std::find(std::begin(*m_buffer_order), std::end(*m_buffer_order), col0);
  auto p1 =
    std::find(std::begin(*m_buffer_order), std::end(*m_buffer_order), col1);
  if (p0 == std::end(*m_buffer_order))
    return p1 != std::end(*m_buffer_order);
  if (p1 == std::end(*m_buffer_order))
    return false;
  return p0 < p1;
}

void
mpims::swap(Reader& r1, Reader& r2) {
  r1.swap(r2);
}
