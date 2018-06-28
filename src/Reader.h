#ifndef READER_H_
#define READER_H_

#include <mpi.h>

#include <complex>
#include <deque>
#include <functional>
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
#include <DataDistribution.h>
#include <GridDistribution.h>
#include <IndexBlock.h>
#include <MSArray.h>
#include <MSColumns.h>
#include <MPIState.h>
#include <TraversalState.h>
#include <ReaderBase.h>
#include <MemoFn.h>

// TODO: unless we make a single, large contiguous datatype for the fileview,
// appending to the file fails in some cases...why?
// #define MPIMS_READER_BIG_CONTIGUOUS_COUNT 10000
#undef MPIMS_READER_BIG_CONTIGUOUS_COUNT

namespace std {

template<>
struct hash<std::vector<mpims::finite_block_t> > {

  std::size_t
  operator() (const std::vector<mpims::finite_block_t>& blks) const {
    std::size_t result = 0;
    for (const auto& blk : blks) {
      std::size_t b0, blen;
      std::tie(b0, blen) = blk;
      result = b0 + 43 * (blen + 43 * result);
    }
    return result;
  };

};

}

namespace mpims {

class IterParams;

template <typename T>
class Writer;

template <typename T>
class Reader
  : public ReaderBase
  , public std::iterator<std::input_iterator_tag, const MSArray<T>, std::size_t> {

  friend class Writer<T>;

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
    MSColumns inner_fileview_axis,
    std::shared_ptr<const ArrayIndexer<MSColumns> > ms_indexer,
    std::size_t buffer_size,
    std::size_t value_extent,
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
  , m_value_extent(value_extent)
  , m_readahead(readahead)
  , m_debug_log(debug_log)
  , m_traversal_state(std::move(traversal_state)) {

    auto handles = m_mpi_state.handles();
    if (handles->comm == MPI_COMM_NULL)
      return;

    MPI_Comm_rank(handles->comm, &m_rank);

    std::array<bool, 2> tests{ m_traversal_state.cont, m_traversal_state.eof() };
    MPI_Allreduce(
      MPI_IN_PLACE,
      tests.data(),
      tests.size(),
      MPI_CXX_BOOL,
      MPI_LAND,
      handles->comm);
    m_traversal_state.cont = tests[0];
    m_traversal_state.global_eof = tests[1];

    advance_to_next_buffer(m_traversal_state, handles->file);
    m_ms_array = read_buffer(m_traversal_state, m_readahead, handles->file);
    if (!m_traversal_state.at_end() && m_readahead)
      start_next();
  }

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

  ~Reader() {
    if (m_readahead) {
      m_ms_array.wait(true);
      m_next_ms_array.wait(true);
    }
  }

  static Reader
  begin(
    const std::string& path,
    const std::string& datarep,
    MPI_Comm comm,
    MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    bool ms_buffer_order,
    const std::unordered_map<MSColumns, GridDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool readahead,
    bool debug_log = false) {
  
    MPI_Comm reduced_comm;
    MPI_File file;
    MPI_Info priv_info;
    std::size_t value_extent;
    std::shared_ptr<std::vector<IterParams> > iter_params;
    std::shared_ptr<std::vector<MSColumns> > buffer_order;
    MSColumns inner_fileview_axis;
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
      value_extent,
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
        value_extent,
        readahead,
        std::move(traversal_state),
        debug_log);
  }

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
  at_end() const {
    return m_traversal_state.at_end();
  }

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
  buffer_length() const {
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
  swap(Reader& other) {
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
    swap(m_value_extent, other.m_value_extent);
    swap(m_ms_array, other.m_ms_array);
    swap(m_next_ms_array, other.m_next_ms_array);
  }

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
  const MSArray<T>&
  operator*() const {
    return getv();
  }

  const MSArray<T>*
  operator->() const {
    return &getv();
  }

protected:

  struct SetFileviewArgs {
    std::unordered_map<MSColumns, std::vector<finite_block_t> > blocks_map;
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
    const std::unordered_map<MSColumns, GridDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool debug_log = false) {
  
    MPI_Comm reduced_comm;
    MPI_File file;
    MPI_Info priv_info;
    std::size_t value_extent;
    std::shared_ptr<std::vector<IterParams> > iter_params;
    std::shared_ptr<std::vector<MSColumns> > buffer_order;
    MSColumns inner_fileview_axis;
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
      ((access_mode == AMode::WriteOnly)
       ? (MPI_MODE_CREATE | MPI_MODE_WRONLY)
       : (MPI_MODE_CREATE | MPI_MODE_RDWR) ),
      reduced_comm,
      priv_info,
      file,
      value_extent,
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
        value_extent,
        false,
        std::move(traversal_state),
        debug_log);
  }

  static void
  initialize(
    const std::string& path,
    const std::string& datarep,
    MPI_Comm comm,
    MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    bool ms_buffer_order,
    const std::unordered_map<MSColumns, GridDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool debug_log,
    int amode,
    MPI_Comm& reduced_comm,
    MPI_Info& priv_info,
    MPI_File& file,
    std::size_t& value_extent,
    std::shared_ptr<std::vector<IterParams> >& iter_params,
    std::shared_ptr<std::vector<MSColumns> >& buffer_order,
    MSColumns &inner_fileview_axis,
    std::shared_ptr<ArrayIndexer<MSColumns> >& ms_indexer,
    std::size_t& buffer_size,
    TraversalState& traversal_state) {

    auto complex_ms_axis =
      std::find_if(
        std::begin(ms_shape),
        std::end(ms_shape),
        [](const auto& ax) {
          return ax.id() == MSColumns::complex;
        });

    bool traversal_has_complex =
      std::any_of(
        std::begin(traversal_order),
        std::end(traversal_order),
        [](const auto& col) {
          return col == MSColumns::complex;
        });

    if (is_complex) {
      if (traversal_has_complex || complex_ms_axis != std::end(ms_shape))
        throw ComplexAxisError();
    } else if (complex_ms_axis != std::end(ms_shape)
               && complex_ms_axis->length() != 2) {
      throw ComplexAxisLengthError();
    }

    // indeterminate axis in MS is allowed only at the outermost axis
    if (
      std::any_of(
        ++std::begin(ms_shape),
        std::end(ms_shape),
        [](const auto& ax) -> bool {
          return !ax.length();
        }))
      throw IndeterminateArrayError();

    // if there is an indeterminate axis, then the traversal order must have the
    // same outermost axis as the MS
    bool indeterminate_ms = !ms_shape[0].length();
    if (indeterminate_ms && traversal_order[0] != (ms_shape)[0].id())
      throw IndeterminateArrayTraversalError();

    buffer_size = (max_buffer_size / value_size) * value_size;

    // complete process grid definition; convert GridDistribution values to
    // DataDistribution values, and create unpartitioned data distributions
    // wherever no explicit grid axis is given
    std::unordered_map<
      MSColumns,
      std::shared_ptr<const DataDistribution> > full_pgrid;
    std::for_each(
      std::begin(ms_shape),
      std::end(ms_shape),
      [&](auto& ax) {
        auto col = ax.id();
        if (pgrid.count(col) > 0)
          full_pgrid[col] = pgrid.at(col)(ax.length());
        else
          full_pgrid[col] = DataDistributionFactory::unpartitioned(ax.length());
      });

    // compute process grid size

    std::size_t pgrid_size = 1;
    std::for_each(
      std::begin(traversal_order),
      std::end(traversal_order),
      [&pgrid_size, &full_pgrid] (const MSColumns& col) {
        pgrid_size *= full_pgrid.at(col)->order();
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
    ms_indexer = ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, ms_shape);

    init_iterparams(ms_shape, traversal_order, full_pgrid, rank, iter_params);

    init_traversal_partitions(
      reduced_comm,
      ms_shape,
      indeterminate_ms,
      buffer_size,
      iter_params,
      inner_fileview_axis,
      value_size,
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
      out << "(" << rank << ") "
          << "m_inner_fileview_axis "
          << mscol_nickname(inner_fileview_axis)
          << std::endl;
      std::clog << out.str();
    }

    auto buffer_dts =
      buffer_datatypes(ms_shape, iter_params, ms_buffer_order, rank, debug_log);

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
    value_extent = value_size;
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
        value_datatype,
        value_datatype,
        datarep.c_str(),
        MPI_INFO_NULL);

      MPI_Aint ve;
      MPI_File_get_type_extent(file, value_datatype, &ve);
      value_extent = static_cast<std::size_t>(ve);
    }

    MemoFn<
      std::vector<finite_block_t>,
      std::shared_ptr<const MPI_Datatype> > fileview_dts =
      fileview_datatypes(
        value_extent,
        ms_shape,
        iter_params,
        rank,
        debug_log);

    // if the MS is indeterminate, then we want to know the initial size of the
    // outermost axis
    MPI_Offset top_len;
    if (indeterminate_ms && reduced_comm != MPI_COMM_NULL) {
      if (rank == 0) {
        // we do this on a single rank and then broadcast the result to minimize
        // the file system metadata operations
        ArrayIndexer<MSColumns>::index index;
        std::for_each(
          std::begin(ms_shape),
          std::end(ms_shape),
          [&index](const auto& ax) {
            index[ax.id()] = 0;
          });
        auto top = ms_shape[0].id();
        std::size_t origin = ms_indexer->offset_of_(index).value();
        ++index[top];
        std::size_t one = ms_indexer->offset_of_(index).value();
        --index[top];
        std::size_t sz = (one - origin) * value_extent;
        MPI_File_get_size(file, &top_len);
        top_len /= sz;
      }
      MPI_Bcast(&top_len, 1, MPI_OFFSET, 0, reduced_comm);
    }

    traversal_state =
      TraversalState(
        reduced_comm,
        iter_params,
        ms_shape[0].id(),
        ((reduced_comm != MPI_COMM_NULL)
         ? ms_shape[0].length().value_or(top_len)
         : 0),
        outer_full_array_axis,
        buffer_dts,
        fileview_dts);
  }

  void
  step(bool cont) {

    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    {
      auto handles = m_mpi_state.handles();
      std::lock_guard<MPIHandles> lock1(*handles);
      if (handles->comm == MPI_COMM_NULL)
        return;
    }
    if (cont && at_end())
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
  start_next() {

    auto handles = m_mpi_state.handles();
    std::lock_guard<MPIHandles> lock1(*handles);
    if (handles->comm == MPI_COMM_NULL)
      return;

    m_next_traversal_state = m_traversal_state;

    bool eof = m_next_traversal_state.eof() || !m_ms_array.buffer();
    std::array<bool, 2> tests{ m_next_traversal_state.cont, eof };
    MPI_Allreduce(
      MPI_IN_PLACE,
      tests.data(),
      tests.size(),
      MPI_CXX_BOOL,
      MPI_LAND,
      handles->comm);
    m_next_traversal_state.cont = tests[0];
    m_next_traversal_state.global_eof = tests[1];

    advance_to_next_buffer(m_next_traversal_state, handles->file);
    m_next_ms_array =
      read_buffer(m_next_traversal_state, m_readahead, handles->file);
  }

  const MSArray<T>&
  getv() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    m_ms_array.wait();
    return m_ms_array;
  };

  // compute datatype for a buffer, assuming data read from file is in MS order
  // (although a fileview will have narrowed the data being read)
  static MemoFn<
    std::size_t,
    std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned> >
  buffer_datatypes(
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::shared_ptr<std::vector<IterParams> >& iter_params,
    bool ms_buffer_order,
    int rank,
    bool debug_log) {

    // the data ordering in the buffer can be different from that in which the
    // data elements are read
    //
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

    auto fn = [=](const std::size_t& buffer_capacity) {

      ArrayIndexer<MSColumns>::index index;
      std::for_each(
        std::begin(*iter_params),
        std::end(*iter_params),
        [&index](const IterParams& ip) {
          index[ip.axis] = 0;
        });

      // compute buffer shape
      std::vector<ColumnAxisBase<MSColumns> > buffer;
      std::for_each(
        std::begin(reordered_ips),
        std::end(reordered_ips),
        [&](auto& ip) {
          if (ip->fully_in_array) {
            // all data on this axis fit into a single buffer
            buffer.emplace_back(
              static_cast<unsigned>(ip->axis),
              ip->size().value());
          } else if (ip->buffer_capacity > 0) {
            // not all of the data on this axis fit into a single buffer
            assert(buffer_capacity <= ip->buffer_capacity);
            buffer.emplace_back(
              static_cast<unsigned>(ip->axis),
              buffer_capacity);
          }
        });

      auto buffer_indexer =
        ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, buffer);

      // build up the buffer datatype by starting at the innermost level of the
      // MS ordering and working outwards; we are creating a possibly transposed
      // array, which requires that we use the hvector datatype
      auto result_dt = datatype(value_datatype);
      unsigned result_dt_count = 1;
      std::for_each(
        ms_shape.crbegin(),
        ms_shape.crend(),
        [&](auto& ax) {
          auto ip = find_iter_params(iter_params, ax.id());

          if (ip->buffer_capacity > 0 && result_dt_count > 1) {
            auto dt = std::move(result_dt);
            result_dt = datatype();
            MPI_Type_contiguous(result_dt_count, *dt, result_dt.get());
            result_dt_count = 1;
          }

          if (ip->fully_in_array || ip->buffer_capacity > 1) {

            auto count =
              (ip->fully_in_array ? ip->size().value() : buffer_capacity);
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
            auto stride = (i1 - i0) * value_size;
            MPI_Aint lb = 0, extent = 0;
            MPI_Type_get_extent(*result_dt, &lb, &extent);
            if (stride == result_dt_count * static_cast<std::size_t>(extent)) {
              result_dt_count *= count;
            } else {
              auto dt = std::move(result_dt);
              result_dt = datatype();
              // use hvector to allow on-the-fly transpositions
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
                  << size / value_size
                  << ", count " << result_dt_count << std::endl;
      }
      if (*result_dt != value_datatype)
        MPI_Type_commit(result_dt.get());
      return std::make_tuple(std::move(result_dt), result_dt_count); 
    };
    return
      MemoFn<
        std::size_t,
        std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned> >(fn);
  }

  static MemoFn<
    std::vector<finite_block_t>,
    std::shared_ptr<const MPI_Datatype> >
  fileview_datatypes(
    MPI_Aint value_extent,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::shared_ptr<std::vector<IterParams> >& iter_params,
    int rank,
    bool debug_log) {

    auto fn = [=](const std::vector<finite_block_t>& fv_blocks) {

      // build datatype for fileview
      std::size_t dt_extent = value_extent;
      auto result = datatype(value_datatype);
      bool last_unbounded = false;

      std::for_each(
        ms_shape.crbegin(),
        ms_shape.crend(),
        [&](auto& ax) {

          auto ip = find_iter_params(iter_params, ax.id());
          std::vector<finite_block_t> blocks;
          if (ip->within_fileview) {
            auto it = ip->begin();
            last_unbounded = it->is_unbounded();
            if (!last_unbounded)
              blocks = it->take_blocked_all();
            else
              blocks = it->take_blocked_while(
                [p=ip->period().value()](auto i){ return i < p; });
          } else if (ip->buffer_capacity > 0) {
            last_unbounded = false;
            if (fv_blocks.size() > 0)
              blocks = fv_blocks;
            else
              blocks.emplace_back(0, ip->buffer_capacity);
          } else {
            last_unbounded = false;
            blocks.emplace_back(0, 1);
          }

          std::size_t len =
            ip->full_fv_axis
            ? ip->axis_length.value_or(ip->period().value())
            : std::max(
              static_cast<decltype(ip->buffer_capacity)>(1),
              ip->buffer_capacity);
          std::tie(result, dt_extent) =
            finite_compound_datatype(
              result,
              dt_extent,
              blocks,
              len,
              rank,
              debug_log);
        });

#ifdef MPIMS_READER_BIG_CONTIGUOUS_COUNT
      if (last_unbounded) {
        auto dt = std::move(result);
        result = datatype();
        MPI_Type_contiguous(
          MPIMS_READER_BIG_CONTIGUOUS_COUNT,
          *dt,
          result.get());
      }
#endif

      MPI_Type_commit(result.get());

      if (debug_log) {
        MPI_Aint lb, extent;
        MPI_Type_get_extent(*result, &lb, &extent);
        MPI_Count size;
        MPI_Type_size_x(*result, &size);
        std::clog << "(" << rank << ") fileview datatype size "
                  << size / value_extent
                  << ", extent " << lb << " " << extent
                  << std::endl;
      }
      return result;
    };
    return
      MemoFn<
        std::vector<finite_block_t>,
        std::shared_ptr<const MPI_Datatype> >(fn);
  }

  MSArray<T>
  read_arrays(
    TraversalState& traversal_state,
    bool nonblocking,
    std::vector<IndexBlockSequence<MSColumns> >& blocks,
    MPI_File file) const {
    // assume that m_mtx and m_mpi_state.handles() are locked

    if (m_debug_log && !traversal_state.at_end()) {
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

    std::unique_ptr<T> buffer;
    if (blocks.size() > 0) {
      if (!traversal_state.at_end())
        buffer.reset(reinterpret_cast<T *>(::operator new(m_buffer_size)));
    } else {
      count = 0;
    }

    if (traversal_state.at_end()) {

      MPI_Status status;
      return MSArray<T>(
        std::move(blocks),
        std::move(buffer),
        std::nullopt,
        count,
        dt,
        std::move(status),
        (m_debug_log ? std::make_optional(m_rank) : std::nullopt)); 

    } else {
      set_deferred_fileview();

      MPI_Offset start;
      if (count > 0)
        MPI_File_get_position(file, &start);
      else
        start = 0;

      if (!nonblocking) {
        MPI_Status status;
        MPI_File_read_all(file, buffer.get(), count, *dt, &status);
        MSArray<T> result(
          std::move(blocks),
          std::move(buffer),
          start,
          count,
          dt,
          std::move(status),
          (m_debug_log ? std::make_optional(m_rank) : std::nullopt));
        result.test_status();
        return result;
      } else {
        MPI_Request request;
        MPI_File_iread_all(file, buffer.get(), count, *dt, &request);
        return MSArray<T>(
          std::move(blocks),
          std::move(buffer),
          start,
          count,
          dt,
          std::move(request),
          (m_debug_log ? std::make_optional(m_rank) : std::nullopt));
      }
    }
  }

  void
  set_fileview(const SetFileviewArgs& args) const {

    // assume that m_mtx and m_mpi_state.handles() are locked

    ArrayIndexer<MSColumns>::index data_index;
    std::for_each(
      std::begin(*m_iter_params),
      std::end(*m_iter_params),
      [&data_index, &args](const auto& ip) {
        auto blks = args.blocks_map.at(ip.axis);
        if (ip.within_fileview || blks.size() == 0)
          data_index[ip.axis] = 0;
        else
          data_index[ip.axis] = std::get<0>(blks[0]);
      });
    std::size_t offset = m_ms_indexer->offset_of_(data_index).value();

    if (m_debug_log) {
      std::ostringstream oss;
      oss << "(" << m_rank << ") fv offset " << offset;
      std::for_each(
        std::begin(*m_iter_params),
        std::end(*m_iter_params),
        [&data_index, &oss](const IterParams& ip) {
          oss << "; " << mscol_nickname(ip.axis) << " " << data_index[ip.axis];
        });
      oss << std::endl;
      std::clog << oss.str();
    }

    m_ms_array.wait();
    m_next_ms_array.wait();

    MPI_File_set_view(
      args.file,
      offset * m_value_extent,
      value_datatype,
      args.datatype,
      m_datarep.c_str(),
      MPI_INFO_NULL);
  }

  void
  set_deferred_fileview() const {
    if (m_deferred_fileview_args) {
      set_fileview(m_deferred_fileview_args.value());
      m_deferred_fileview_args.reset();
    }
  }

  void
  advance_to_next_buffer(
    TraversalState& traversal_state,
    MPI_File file) const {

    // assume that m_mtx and m_mpi_state.handles() are locked

    traversal_state.advance_to_next_buffer(
      m_inner_fileview_axis,
      [this, &traversal_state, &file]() {

        auto args = SetFileviewArgs {
          traversal_state.blocks_map(),
          *traversal_state.fileview_datatype(m_inner_fileview_axis),
          file
        };

        if (traversal_state.at_end())
          m_deferred_fileview_args = std::make_optional(args);
        else
          set_fileview(args);
      });
  }

  void
  extend() {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    m_traversal_state.global_eof = false;
  }

  MSArray<T>
  read_buffer(
    TraversalState& traversal_state,
    bool nonblocking,
    MPI_File file) const {

    // assume that m_mtx is locked
    auto blocks = sorted_blocks(traversal_state);
    auto result = read_arrays(traversal_state, nonblocking, blocks, file);
    traversal_state.advance_to_buffer_end();
    return result;
  }

  mutable std::shared_ptr<const MPI_Datatype> m_buffer_datatype;

  bool
  buffer_order_compare(const MSColumns& col0, const MSColumns& col1) const {
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

  std::vector<IndexBlockSequence<MSColumns> >
  sorted_blocks(const TraversalState& state) const {
    auto result = state.blocks();
    std::stable_sort(
      std::begin(result),
      std::end(result),
      [this](const IndexBlockSequence<MSColumns>& ibs0,
             const IndexBlockSequence<MSColumns>& ibs1) {
        return buffer_order_compare(ibs0.m_axis, ibs1.m_axis);
      });
    return result;
  }

  static MPI_Datatype value_datatype;

  static std::size_t value_size;

  static bool is_complex;

private:

  MPIState m_mpi_state;

  std::string m_datarep;

  std::shared_ptr<const std::vector<ColumnAxisBase<MSColumns> > > m_ms_shape;

  // ms array iteration definition, for this rank, in traversal order
  std::shared_ptr<const std::vector<IterParams> > m_iter_params;

  std::shared_ptr<const std::vector<MSColumns> > m_buffer_order;

  MSColumns m_inner_fileview_axis;

  std::shared_ptr<const MPI_Datatype> m_etype_datatype;

  std::shared_ptr<const ArrayIndexer<MSColumns> > m_ms_indexer;

  int m_rank;

  std::size_t m_buffer_size;

  std::size_t m_value_extent;

  bool m_readahead;

  bool m_debug_log;

  TraversalState m_traversal_state;

  TraversalState m_next_traversal_state;

  mutable MSArray<T> m_ms_array;

  mutable MSArray<T> m_next_ms_array;

  mutable std::recursive_mutex m_mtx;

  mutable std::optional<SetFileviewArgs> m_deferred_fileview_args;
};

template <typename T>
void
swap(Reader<T>& r1, Reader<T>& r2) {
  r1.swap(r2);
}

using CxFltReader = Reader<std::complex<float> >;
template <> MPI_Datatype CxFltReader::value_datatype;
template <> std::size_t CxFltReader::value_size;
template <> bool CxFltReader::is_complex;

using FltReader = Reader<float>;
template <> MPI_Datatype FltReader::value_datatype;
template <> std::size_t FltReader::value_size;
template <> bool FltReader::is_complex;

using CxDblReader = Reader<std::complex<double> >;
template <> MPI_Datatype CxDblReader::value_datatype;
template <> std::size_t CxDblReader::value_size;
template <> bool CxDblReader::is_complex;

using DblReader = Reader<double>;
template <> MPI_Datatype DblReader::value_datatype;
template <> std::size_t DblReader::value_size;
template <> bool DblReader::is_complex;

} // end namespace mpims

#endif // READER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
