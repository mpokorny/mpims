/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <Reader.h>

using namespace mpims;

// #define USE_INDEXED

Reader::Reader()
  : m_readahead(false) {
}

Reader::Reader(
  MPIState&& mpi_state,
  const std::string& datarep,
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
  , m_datarep(datarep)
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
  if (handles->comm == MPI_COMM_NULL)
    return;

  MPI_Comm_rank(handles->comm, reinterpret_cast<int*>(&m_rank));

  if (handles->file != MPI_FILE_NULL) {
    if (!*m_inner_fileview_axis)
      set_fileview(m_traversal_state, handles->file);
    const IterParams* init_params = &(*m_iter_params)[0];
    m_traversal_state.axis_iters.emplace(
      std::shared_ptr<const IterParams>(m_iter_params, init_params),
      !init_params->max_blocks || init_params->max_blocks > 0);
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
    m_ms_array.wait(true);
    m_next_ms_array.wait(true);
  }
}

Reader
Reader::begin(
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
  bool debug_log) {

  MPI_Comm reduced_comm;
  MPI_File file;
  MPI_Info priv_info;
  std::shared_ptr<std::vector<IterParams> > iter_params;
  std::shared_ptr<std::vector<MSColumns> > buffer_order;
  std::shared_ptr<std::optional<MSColumns> > inner_fileview_axis;
  std::shared_ptr<ArrayIndexer<MSColumns> > ms_indexer;
  std::size_t buffer_size;
  TraversalState traversal_state;

  initialize(
    path,
    datarep,
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
      datarep,
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
  const std::string& datarep,
  MPI_Comm comm,
  MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  bool ms_buffer_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t max_buffer_size,
  bool readahead,
  bool debug_log) {

  MPI_Comm reduced_comm;
  MPI_File file;
  MPI_Info priv_info;
  std::shared_ptr<std::vector<IterParams> > iter_params;
  std::shared_ptr<std::vector<MSColumns> > buffer_order;
  std::shared_ptr<std::optional<MSColumns> > inner_fileview_axis;
  std::shared_ptr<ArrayIndexer<MSColumns> > ms_indexer;
  std::size_t buffer_size;
  TraversalState traversal_state;

  initialize(
    path,
    datarep,
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

  MPI_File_set_atomicity(file, true); // TODO: not needed for write only

  return
    Reader(
      MPIState(reduced_comm, priv_info, file, path),
      datarep,
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
    pdist.block_size =
      std::min(ax->length().value_or(SIZE_MAX), pdist.block_size);
    pdist.num_processes =
      (ax->is_unbounded()
       ? pdist.num_processes
       : std::min(ceil(ax->length().value(), pdist.block_size),
                  pdist.num_processes));
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
  MPI_Comm_size(comm, &comm_size);
  if (static_cast<std::size_t>(comm_size) < pgrid_size)
    throw std::runtime_error("too few processes for grid");

  // create a new communicator of the minimum needed size
  if (static_cast<std::size_t>(comm_size) > pgrid_size) {
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_split(
      comm,
      ((static_cast<std::size_t>(comm_rank) < pgrid_size) ? 1 : MPI_UNDEFINED),
      comm_rank,
      &reduced_comm);
  } else {
    MPI_Comm_dup(comm, &reduced_comm);
  }
  int rank = 0;
  if (reduced_comm != MPI_COMM_NULL) {
    set_throw_exception_errhandler(reduced_comm);
    MPI_Comm_rank(reduced_comm, &rank);
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

  std::shared_ptr<MPI_Datatype> full_buffer_datatype;
  unsigned full_buffer_dt_count;
  std::tie(full_buffer_datatype, full_buffer_dt_count) =
    init_buffer_datatype(
      ms_shape,
      iter_params,
      ms_buffer_order,
      false,
      rank,
      debug_log);

  std::shared_ptr<MPI_Datatype> tail_buffer_datatype;
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
    MPI_Info_dup(info, &priv_info);
  if (reduced_comm != MPI_COMM_NULL) {
    MPI_File_open(reduced_comm, path.c_str(), amode, priv_info, &file);
    set_throw_exception_errhandler(file);
    if (debug_log && rank == 0) {
      MPI_Info info_used;
      MPI_File_get_info(file, &info_used);
      std::array<char,80> value;
      int flag;
      MPI_Info_get(
        info_used,
        "romio_filesystem_type",
        value.size() - 1,
        value.data(),
        &flag);
      if (flag != 0)
        std::clog << "ROMIO filesystem type: " << value.data() << std::endl;
      else
        std::clog << "No ROMIO filesystem type" << std::endl;
      MPI_Info_free(&info_used);
    }
    // set the view here in order to associate a data representation with the
    // file handle
    MPI_File_set_view(
      file,
      0,
      MPI_CXX_FLOAT_COMPLEX,
      MPI_CXX_FLOAT_COMPLEX,
      datarep.c_str(),
      MPI_INFO_NULL);
  }

  std::shared_ptr<MPI_Datatype> full_fileview_datatype =
    init_fileview(
      file,
      ms_shape,
      iter_params,
      ms_indexer,
      false,
      rank,
      debug_log);

  std::shared_ptr<MPI_Datatype> tail_fileview_datatype =
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
    m_ms_array.wait(!cont);
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

  // TODO: modify for unbounded ms
  m_next_traversal_state.eof =
    m_next_traversal_state.axis_iters.empty();
  std::array<bool, 2>
    tests{ m_next_traversal_state.cont, m_next_traversal_state.eof };
  auto handles = m_mpi_state.handles();
  std::lock_guard<MPIHandles> lock1(*handles);
  MPI_Allreduce(
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
    m_ms_array.wait();
    m_next_ms_array =
      read_next_buffer(m_next_traversal_state, m_readahead, handles->file);
  } else {
    m_next_ms_array = MSArray();
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
      result *= std::get<0>(ibs.num_elements());
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
  swap(m_datarep, other.m_datarep);
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
      auto length = ax.length();
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
      assert(!length || block_len <= length.value());
      std::size_t origin = order * block_len;
      std::size_t stride = grid_len * block_len;
      std::optional<std::size_t> max_blocks =
        (length
         ? std::optional<std::size_t>(ceil(length.value(), stride))
         : std::nullopt);
      std::size_t blocked_rem =
        (length
         ? (ceil(length.value(), block_len) % grid_len)
         : 0);
      std::size_t terminal_block_len;
      std::size_t max_terminal_block_len;
      if (blocked_rem == 0) {
        terminal_block_len = block_len;
        max_terminal_block_len = block_len;
      } else if (blocked_rem == 1) {
        // assert(length.has_value());
        terminal_block_len = ((order == 0) ? (length.value() % block_len) : 0);
        max_terminal_block_len = length.value() % block_len;
      } else {
        // assert(length.has_value());
        terminal_block_len =
          ((order < blocked_rem - 1)
           ? block_len
           : ((order == blocked_rem - 1) ? (length.value() % block_len) : 0));
        max_terminal_block_len = block_len;
      }
      (*iter_params)[traversal_index] =
        IterParams { col, false, false, 0, array_length,
                     origin, stride, block_len, terminal_block_len,
                     max_terminal_block_len, length, max_blocks };
      if (debug_log) {
        auto l = (*iter_params)[traversal_index].length;
        auto mb = (*iter_params)[traversal_index].max_blocks;
        std::clog << "(" << rank << ") "
                  << mscol_nickname(col)
                  << " length: "
                  << (l ? std::to_string(l.value()) : "unbounded")
                  << ", origin: "
                  << (*iter_params)[traversal_index].origin
                  << ", stride: "
                  << (*iter_params)[traversal_index].stride
                  << ", block_len: "
                  << (*iter_params)[traversal_index].block_len
                  << ", max_blocks: "
                  << (mb ? std::to_string(mb.value()) : "unbounded")
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
    std::size_t len =
      std::min(
        (max_buffer_length / start_buffer->array_length)
        / start_buffer->block_len,
        start_buffer->max_blocks.value_or(SIZE_MAX))
      * start_buffer->block_len;
    std::optional<std::size_t> full_len = start_buffer->max_accessible_length();
    if (len == 0) {
      if (start_buffer != iter_params->rbegin()) {
        --start_buffer;
        // assert(start_buffer->max_blocks);
        start_buffer->buffer_capacity =
          start_buffer->max_blocks.value() * start_buffer->block_len;
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
    if (!full_len || len < full_len.value())
      break;
    array_length *= full_len.value();
    ++start_buffer;
  }
  if (start_buffer == iter_params->rend()) {
    --start_buffer;
    // assert(start_buffer->max_blocks);
    start_buffer->buffer_capacity =
      start_buffer->max_blocks.value() * start_buffer->block_len;
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
        if ((ooo
             && !(ip->fully_in_array
                  || (ip->max_blocks
                      && (ip->buffer_capacity >= ip->max_accessible_length()))))
            || !std::get<1>(tail_buffer_blocks(*ip)))
          *inner_fileview_axis = ip->axis;
      }
      ip->within_fileview = !inner_fileview_axis->has_value();
      ++ip;
    }
  }
}

std::tuple<std::unique_ptr<MPI_Datatype, DatatypeDeleter>, unsigned>
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
      auto accessible_length = ip->accessible_length();
      if (ip->fully_in_array) {
        buffer.emplace_back(
          static_cast<unsigned>(ip->axis),
          accessible_length.value());
      } else if (ip->buffer_capacity > 0) {
        buffer_capacity = ip->buffer_capacity;
        tail_buffer_capacity = (
          accessible_length
          ? (accessible_length.value() % ip->buffer_capacity)
          : 0);
        std::size_t capacity =
          tail_array ? tail_buffer_capacity : buffer_capacity;
        buffer.emplace_back(static_cast<unsigned>(ip->axis), capacity);
      }
    });

  if (tail_array) {
    if (tail_buffer_capacity == 0)
      return std::make_tuple(
        std::unique_ptr<MPI_Datatype, DatatypeDeleter>(),
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
        auto count =
          (ip->fully_in_array
           ? ip->accessible_length().value()
           : buffer_capacity);
        auto i0 = buffer_indexer->offset_of_(index).value();
        ++index[ip->axis];
        auto i1 = buffer_indexer->offset_of_(index).value();
        --index[ip->axis];
        if (debug_log) {
          std::clog << "(" << rank << ") "
                    << mscol_nickname(ip->axis)
                    << " dv stride " << i1 - i0
                    << std::endl;
        }
        auto stride = (i1 - i0) * sizeof(std::complex<float>);
        MPI_Aint lb = 0, extent = 0;
        MPI_Type_get_extent(*result_dt, &lb, &extent);
        if (stride == result_dt_count * static_cast<std::size_t>(extent)) {
          result_dt_count *= count;
        } else {
          auto dt = std::move(result_dt);
          result_dt = datatype();
          MPI_Type_create_hvector(
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
    MPI_Count size;
    MPI_Type_size_x(*result_dt, &size);
    std::clog << "(" << rank << ") buffer datatype size "
              << size / sizeof(std::complex<float>)
              << ", count " << result_dt_count << std::endl;
  }
  if (*result_dt != MPI_CXX_FLOAT_COMPLEX)
    MPI_Type_commit(result_dt.get());
  return std::make_tuple(std::move(result_dt), result_dt_count);
}

std::tuple<
  std::unique_ptr<MPI_Datatype, DatatypeDeleter>,
  std::size_t>
Reader::vector_datatype(
  std::size_t value_extent,
  const std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
  std::size_t dt_extent,
  std::size_t num_blocks,
  std::size_t block_len,
  std::size_t terminal_block_len,
  std::size_t stride,
  std::size_t len,
  int rank,
  bool debug_log) {

  if (debug_log) {
    std::ostringstream oss;
    oss << "(" << rank << ") "
        << "vector_datatype(ve " << value_extent
        << ", de " << dt_extent
        << ", nb " << num_blocks
        << ", bl " << block_len
        << ", tb " << terminal_block_len
        << ", st " << stride
        << ", ln " << len
        << ")" << std::endl;
    std::clog << oss.str();
  }
  auto result_dt = datatype();
  std::size_t result_dt_extent = len * dt_extent;
  std::size_t size = result_dt_extent * value_extent;
  auto nb = num_blocks - ((terminal_block_len == 0) ? 1 : 0);
  if (nb * block_len > 1) {
    auto dt1 = datatype();
    if (block_len == terminal_block_len || terminal_block_len == 0) {
      if (block_len == stride)
        MPI_Type_contiguous(nb * block_len, *dt, dt1.get());
      else
        MPI_Type_vector(nb, block_len, stride, *dt, dt1.get());
    } else {
      auto blocklengths = std::make_unique<int[]>(nb);
      auto displacements = std::make_unique<int[]>(nb);
      for (std::size_t i = 0; i < nb; ++i) {
        blocklengths[i] = block_len;
        displacements[i] = i * stride;
      }
      blocklengths[nb - 1] = terminal_block_len;
      MPI_Type_indexed(
        nb,
        blocklengths.get(),
        displacements.get(),
        *dt,
        dt1.get());
    }
    MPI_Type_create_resized(*dt1, 0, size, result_dt.get());
  } else {
    MPI_Type_create_resized(*dt, 0, size, result_dt.get());
  }
  return std::make_tuple(std::move(result_dt), result_dt_extent);
}

std::tuple<std::unique_ptr<MPI_Datatype, DatatypeDeleter>, std::size_t, bool>
Reader::compound_datatype(
  std::size_t value_extent,
  const std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
  std::size_t dt_extent,
  std::size_t stride,
  std::size_t num_blocks,
  std::size_t block_len,
  std::size_t terminal_block_len,
  const std::optional<std::size_t>& len,
  int rank,
  bool debug_log) {

  // resize current fileview_datatype to equal stride between elements
  // on this axis
  std::unique_ptr<MPI_Datatype, DatatypeDeleter> result_dt;
  std::size_t result_dt_extent;

  // create (blocked) vector of fileview_datatype elements
  std::tie(result_dt, result_dt_extent) =
    vector_datatype(
      value_extent,
      dt,
      dt_extent,
      num_blocks,
      block_len,
      terminal_block_len,
      stride,
      len.value_or(stride),
      rank,
      debug_log);
  return
    std::make_tuple(std::move(result_dt), result_dt_extent, !len.has_value());
}

std::tuple<std::size_t, bool>
Reader::tail_buffer_blocks(const IterParams& ip) {
  std::size_t tail_num_blocks;
  bool uniform;
  if (!ip.fully_in_array) {
    std::size_t capacity = std::max(ip.buffer_capacity, 1uL);
    auto nr = ip.stride / ip.block_len;
    auto full_buffer_capacity = capacity * nr;
    auto nb = ceil(ip.length.value(), full_buffer_capacity);
    auto tail_origin = full_buffer_capacity * (nb - 1);
    auto tail_rem = ip.length.value() - tail_origin;
    tail_num_blocks = ceil(tail_rem, ip.stride);
    uniform = tail_rem % ip.stride == 0;
  } else {
    tail_num_blocks = 0;
    uniform = true;
  }
  return std::make_tuple(tail_num_blocks, uniform);
}

std::unique_ptr<MPI_Datatype, DatatypeDeleter>
Reader::init_fileview(
  MPI_File file,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::shared_ptr<std::vector<IterParams> >& iter_params,
  const std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
  bool tail_fileview,
  int rank,
  bool debug_log) {

  ArrayIndexer<MSColumns>::index index;
  std::size_t tail_num_blocks;
  std::for_each(
    ms_shape.crbegin(),
    ms_shape.crend(),
    [&](auto& ax) {
      auto ip = find_iter_params(iter_params, ax.id());
      index[ip->axis] = 0;
      if (!ip->within_fileview && ip->buffer_capacity > 0) {
        std::tie(tail_num_blocks, std::ignore) =
          tail_buffer_blocks(*ip);
      }
    });

  // build datatype for fileview
  auto result = datatype(MPI_CXX_FLOAT_COMPLEX);
  MPI_Aint fcx_extent;
  MPI_File_get_type_extent(file, *result, &fcx_extent);
  std::size_t value_extent = static_cast<std::size_t>(fcx_extent);
  bool unbounded_dt_count = false;
  std::size_t dt_extent = 1;

  std::for_each(
    ms_shape.crbegin(),
    ms_shape.crend(),
    [&](auto& ax) {

      if (unbounded_dt_count)
        throw std::runtime_error("unbounded");

      auto ip = find_iter_params(iter_params, ax.id());
      std::size_t num_blocks = 1;
      std::optional<std::size_t> len = ip->length;
      std::size_t block_len = ip->block_len;
      std::size_t terminal_block_len = ip->terminal_block_len;
      if (ip->within_fileview) {
        num_blocks = ip->max_blocks.value();
      } else if (ip->buffer_capacity > 0) {
        if (!tail_fileview) {
          assert(ip->buffer_capacity % ip->block_len == 0);
          num_blocks = ip->buffer_capacity / ip->block_len;
          terminal_block_len = ip->block_len;
        } else {
          num_blocks = tail_num_blocks;
        }
      }
      auto i0 = ms_indexer->offset_of_(index).value();
      ++index[ip->axis];
      auto i1 = ms_indexer->offset_of_(index).value();
      --index[ip->axis];
      std::size_t unit_stride = i1 - i0;
      index[ip->axis] += ip->stride;
      auto is = ms_indexer->offset_of_(index).value();
      index[ip->axis] -= ip->stride;
      std::size_t block_stride = is - i0;

      std::tie(result, dt_extent, unbounded_dt_count) =
        compound_datatype(
          value_extent,
          result,
          dt_extent,
          block_stride / unit_stride,
          num_blocks,
          block_len,
          terminal_block_len,
          len,
          rank,
          debug_log);
    });

  MPI_Type_commit(result.get());

  if (debug_log) {
    MPI_Count size;
    MPI_Type_size_x(*result, &size);
    std::clog << "(" << rank << ") fileview datatype size "
              << size / value_extent
              << std::endl;
  }
  return result;
}


void
Reader::set_fileview(TraversalState& traversal_state, MPI_File file) const {
  // assume that m_mtx and m_mpi_state.handles() are locked
  std::size_t offset =
    m_ms_indexer->offset_of_(traversal_state.data_index).value();

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
    oss << (traversal_state.in_tail ? "" : " not") << " in tail";
    oss << std::endl;
    std::clog << oss.str();
  }

  const MPI_Datatype dt = *traversal_state.fileview_datatype();

  MPI_File_set_view(
    file,
    offset * sizeof(std::complex<float>),
    MPI_CXX_FLOAT_COMPLEX,
    dt,
    m_datarep.c_str(),
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
        auto accessible_length = ip.accessible_length();
        std::vector<IndexBlock> merged_blocks;
        if (accessible_length) {
          // merge contiguous blocks
          std::size_t start = ip.origin;
          std::size_t end = ip.origin + ip.block_len;
          for (std::size_t b = 1; b < ip.max_blocks.value(); ++b) {
            std::size_t s = ip.origin + b * ip.stride;
            if (s > end) {
              merged_blocks.emplace_back(start, end - start);
              start = s;
            }
            end = s + ip.block_len;
          }
          end -= ip.block_len - ip.terminal_block_len;
          merged_blocks.emplace_back(start, end - start);
        } else {
          // create enough blocks to fit buffer_capacity, merging contiguous
          // blocks
          assert(ip.buffer_capacity % ip.block_len == 0);
          auto nb = ip.buffer_capacity / ip.block_len;
          auto stride = nb * ip.stride;
          std::size_t start = ip.origin;
          std::size_t end = ip.origin + ip.stride;
          for (std::size_t b = 1; b < nb; ++b) {
            std::size_t s = start + b * ip.stride;
            if (s > end) {
              merged_blocks.emplace_back(start, end - start, stride);
              start = s;
            }
            end = s + ip.block_len;
          }
          merged_blocks.emplace_back(start, end - start, stride);
        }

        // when entire axis doesn't fit into the array, we might have to split
        // blocks
        if (!ip.fully_in_array
            && (accessible_length
                && ip.accessible_length().value() > ip.buffer_capacity)) {
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

MSArray
Reader::read_arrays(
  TraversalState& traversal_state,
  bool nonblocking,
  std::vector<IndexBlockSequence<MSColumns> >& blocks,
  MPI_File file) const {
  // assume that m_mtx and m_mpi_state.handles() are locked

  if (m_debug_log) {
    std::ostringstream oss;
    oss << "(" << m_rank << ") read ";
    const char* sep0 = "";
    std::for_each(
      std::begin(blocks),
      std::end(blocks),
      [&sep0,&oss](auto& ibs) {
        oss << sep0
                  << mscol_nickname(ibs.m_axis) << ": [";
        sep0 = "; ";
        const char *sep1 = "";
        std::for_each(
          std::begin(ibs.m_blocks),
          std::end(ibs.m_blocks),
          [&sep1,&oss](auto& b) {
            oss << sep1 << "(" << b.m_index << "," << b.m_length << ")";
            sep1 = ",";
          });
        oss << "]";
      });
    std::clog << oss.str() << std::endl;
  }

  std::shared_ptr<const MPI_Datatype> dt;
  unsigned count;
  std::tie(dt, count) = traversal_state.buffer_datatype();

  std::unique_ptr<std::complex<float> > buffer;
  if (traversal_state.count > 0)
    buffer.reset(
      reinterpret_cast<std::complex<float> *>(::operator new(m_buffer_size)));
  else
    count = 0;

  MPI_Offset start;
  MPI_File_get_position(file, &start);

  if (!nonblocking) {
    MPI_Status status;
    MPI_File_read_all(file, buffer.get(), count, *dt, &status);
    MSArray result(
      std::move(blocks),
      std::move(buffer),
      start,
      count,
      dt,
      std::move(status),
      (m_debug_log ? std::optional<int>(m_rank) : std::nullopt));
    result.test_status();
    return result;
  } else {
    MPI_Request request;
    MPI_File_iread_all(file, buffer.get(), count, *dt, &request);
    return MSArray(
      std::move(blocks),
      std::move(buffer),
      start,
      count,
      dt,
      std::move(request),
      (m_debug_log ? std::optional<int>(m_rank) : std::nullopt));
  }
}

void
Reader::advance_to_next_buffer(
  TraversalState& traversal_state,
  MPI_File file) const {
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
        auto nr = axis_iter.num_remaining();
        if (nr
            && (nr.value()
                < static_cast<std::size_t>(traversal_state.max_count))) {
          traversal_state.count = nr.value();
          traversal_state.in_tail = true;
        } else {
          traversal_state.count = traversal_state.max_count;
          traversal_state.in_tail = false;
        }
      } else {
        traversal_state.in_tail = false;
      }
      if (*m_inner_fileview_axis && axis == m_inner_fileview_axis->value()) {
        m_ms_array.wait();
        m_next_ms_array.wait();
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

MSArray
Reader::read_next_buffer(
  TraversalState& traversal_state,
  bool nonblocking,
  MPI_File file) const {

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
  return read_arrays(traversal_state, nonblocking, blocks, file);
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
