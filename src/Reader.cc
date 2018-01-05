/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <stack>
#include <stdexcept>
#include <unordered_set>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <Reader.h>

using namespace mpims;

Reader::Reader(
  const std::string& path,
  ::MPI_Comm comm,
  ::MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  std::unordered_map<MSColumns, ProcessDistribution>& pgrid,
  std::size_t max_buffer_size)
  : m_ms_shape(ms_shape)
  , m_comm(MPI_COMM_NULL)
  , m_file(MPI_FILE_NULL)
  , m_array_datatype_predef(false)
  , m_array_datatype(MPI_DATATYPE_NULL)
  , m_buffer_size(
    (max_buffer_size / sizeof(std::complex<float>))
    * sizeof(std::complex<float>))
  , m_fileview_datatype_predef(false)
  , m_fileview_datatype(MPI_DATATYPE_NULL)
  , m_ms_indexer(
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, m_ms_shape)) {

  // reduce process grid for small (blocked) axis sizes
  for (auto& pg : pgrid) {
    MSColumns pcol;
    ProcessDistribution pdist;
    std::tie(pcol, pdist) = pg;
    auto ax =
      std::find_if(
        std::begin(m_ms_shape),
        std::end(m_ms_shape),
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
      &m_comm);
    if (m_comm == MPI_COMM_NULL)
      return;
  } else {
    mpi_call(::MPI_Comm_dup, comm, &m_comm);
  }
  mpi_call(::MPI_Comm_rank, m_comm, &m_rank);

  init_iterparams(traversal_order, pgrid);
  init_outer_array_axis();
  init_fileview();
  adjust_outer_array_axis();
  // std::cout << "m_outer_array_axis " << mscol_nickname(m_outer_array_axis) << std::endl;
  // if (m_inner_fileview_axis)
  //   std::cout << "m_inner_fileview_axis " << mscol_nickname(m_inner_fileview_axis.value()) << std::endl;
  // else
  //   std::cout << "no m_inner_fileview_axis" << std::endl;
  init_array_datatype();

  mpi_call(
    ::MPI_File_open,
    m_comm,
    path.c_str(),
    MPI_MODE_RDONLY,
    info,
    &m_file);
  mpi_call(::MPI_File_set_errhandler, m_file, MPI_ERRORS_RETURN);
}

void
Reader::finalize() {
  if (m_file != MPI_FILE_NULL) {
    try {
      mpi_call(::MPI_File_close, &m_file);
    } catch (const mpi_error& e) {
      std::cerr << "MPI_File_close failed in ~Reader(): "
                << e.what()
                << std::endl;
    }
  }
  if (m_array_datatype != MPI_DATATYPE_NULL
      && !m_array_datatype_predef) {
    try {
      mpi_call(::MPI_Type_free, &m_array_datatype);
    } catch (const mpi_error& e) {
      std::cerr << "MPI_Type_free failed in ~Reader(): "
                << e.what()
                << std::endl;
    }
  }
  if (m_fileview_datatype != MPI_DATATYPE_NULL
      && !m_fileview_datatype_predef) {
    try {
      mpi_call(::MPI_Type_free, &m_fileview_datatype);
    } catch (const mpi_error& e) {
      std::cerr << "MPI_Type_free failed in ~Reader(): "
                << e.what()
                << std::endl;
    }
  }
  if (m_comm != MPI_COMM_NULL) {
    try {
      mpi_call(::MPI_Comm_free, &m_comm);
    } catch (const mpi_error& e) {
      std::cerr << "MPI_Comm_free failed in ~Reader(): "
                << e.what()
                << std::endl;
    }
  }
}

void
Reader::init_iterparams(
  const std::vector<MSColumns>& traversal_order,
  const std::unordered_map<MSColumns, ProcessDistribution>& pgrid) {

  m_iter_params.resize(traversal_order.size());
  std::size_t dist_size = 1;
  std::for_each(
    std::begin(m_ms_shape),
    std::end(m_ms_shape),
    [this, &dist_size, &traversal_order, &pgrid]
    (const ColumnAxisBase<MSColumns>& ax) mutable {
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
      std::size_t order = (m_rank / dist_size) % grid_len;
      std::size_t block_len = block_size;
      assert(block_len <= length);
      std::size_t origin = order * block_len;
      std::size_t stride = grid_len * block_len;
      std::size_t max_blocks = (length + stride - 1) / stride;
      m_iter_params[traversal_index] =
        IterParams { col, false, num_processes > 1, length, origin, stride,
                     block_len, max_blocks };
      dist_size *= grid_len;
    });
}

void
Reader::init_outer_array_axis() {
  // compute shape of largest array with full-length dimensions (for this rank)
  // that buffer of requested size can contain, building up from innermost
  // traversal axis
  std::size_t buffer_length = m_buffer_size / sizeof(std::complex<float>);
  std::size_t array_length = 1;
  auto start_array = m_iter_params.rbegin();
  auto next_array_length =
    array_length * start_array->block_len * start_array->max_blocks;
  while (start_array != m_iter_params.rend()
         && next_array_length <= buffer_length) {
    array_length = next_array_length;
    start_array->in_array = true;
    ++start_array;
    if (start_array != m_iter_params.rend())
      next_array_length =
        array_length * start_array->block_len * start_array->max_blocks;
  }
  if (start_array == m_iter_params.rbegin())
    throw std::runtime_error("maximum buffer size too small");
  --start_array;

  m_outer_array_axis = start_array->axis;
}

void
Reader::init_array_datatype() {

  ArrayIndexer<MSColumns>::index index;
  std::for_each(
    std::begin(m_iter_params),
    std::end(m_iter_params),
    [&index](const IterParams& ip) {
      index[ip.axis] = 0;
    });

  std::vector<ColumnAxisBase<MSColumns> > array;
  std::for_each(
    std::begin(m_iter_params),
    std::end(m_iter_params),
    [&array](const IterParams& ip) {
      if (ip.in_array)
        array.emplace_back(
          static_cast<unsigned>(ip.axis),
          ip.block_len * ip.max_blocks);
    });
  auto array_indexer =
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, array);

  m_array_datatype = MPI_CXX_FLOAT_COMPLEX;
  m_array_datatype_predef = true;
  std::for_each(
    m_ms_shape.crbegin(),
    m_ms_shape.crend(),
    [this, &array_indexer, &index](const ColumnAxisBase<MSColumns>& ax) {
      auto ax_id = ax.id();
      auto array_iter_param =
        std::find_if(
          std::begin(m_iter_params),
          std::end(m_iter_params),
          [&ax_id](const IterParams& ip) {
            return ip.axis == ax_id;
          });
      if (array_iter_param->in_array) {
        auto count = array_iter_param->block_len * array_iter_param->max_blocks;
        if (count > 1) {
          auto i0 = array_indexer->offset_of_(index);
          index[ax_id] = 1;
          auto i1 = array_indexer->offset_of_(index);
          index[ax_id] = 0;
          auto stride = (i1 - i0) * sizeof(std::complex<float>);
          ::MPI_Datatype dt = m_array_datatype;
          mpi_call(
            ::MPI_Type_create_hvector,
            count,
            1,
            stride,
            dt,
            &m_array_datatype);
          if (!m_array_datatype_predef)
            mpi_call(::MPI_Type_free, &dt);
          m_array_datatype_predef = false;
        }
      }
    });

  if (!m_array_datatype_predef)
    mpi_call(::MPI_Type_commit, &m_array_datatype);
}

void
Reader::init_fileview() {

  // find innermost out-of-order traversal axis
  std::unordered_set<MSColumns> out_of_order;
  auto ms_axis = m_ms_shape.crbegin();
  auto ip = m_iter_params.crbegin();
  while (ms_axis != m_ms_shape.crend()) {
    if (ms_axis->id() != ip->axis)
      out_of_order.insert(ms_axis->id());
    else
      ++ip;
    ++ms_axis;
  }
  ip = m_iter_params.crbegin();
  while (!m_inner_fileview_axis && ip != m_iter_params.crend()) {
    // select the innermost of the out of order traversal axes that is not among
    // the array axes
    if (!ip->in_array && out_of_order.count(ip->axis) > 0)
      m_inner_fileview_axis = ip->axis;
    ++ip;
  }
  // now move m_inner_fileview_axis inwards until it's not above any distributed
  // axes
  bool below_inner_fileview = !m_inner_fileview_axis.has_value();
  auto ipf = std::begin(m_iter_params);
  while (ipf != std::end(m_iter_params) && !below_inner_fileview) {
    below_inner_fileview = ipf->axis == m_inner_fileview_axis.value();
    ++ipf;
  }
  while (ipf != std::end(m_iter_params)) {
    if (ipf->is_distributed)
      m_inner_fileview_axis = ipf->axis;
    ++ipf;
  }

  // build datatype for fileview
  m_fileview_datatype = MPI_CXX_FLOAT_COMPLEX;
  m_fileview_datatype_predef = true;
  if (m_inner_fileview_axis) {

    ArrayIndexer<MSColumns>::index index;
    std::for_each(
      std::begin(m_iter_params),
      std::end(m_iter_params),
      [&index](const IterParams& ip) {
        index[ip.axis] = 0;
      });

    ms_axis = m_ms_shape.crbegin();
    ip = m_iter_params.crbegin();
    while (ip->axis != m_inner_fileview_axis.value()) {
      auto ms_axis_id = ms_axis->id();
      ::MPI_Datatype dt1 = m_fileview_datatype;
      if (ms_axis_id == ip->axis) {
        auto count = ip->block_len * ip->max_blocks;
        if (count > 1) {
          auto i0 = m_ms_indexer->offset_of_(index);
          index[ip->axis] = ip->stride;
          auto i1 = m_ms_indexer->offset_of_(index);
          index[ip->axis] = 0;
          auto stride = (i1 - i0) * sizeof(std::complex<float>);
          mpi_call(
            ::MPI_Type_create_hvector,
            ip->max_blocks,
            ip->block_len,
            stride,
            dt1,
            &m_fileview_datatype);
          if (!m_fileview_datatype_predef)
            mpi_call(::MPI_Type_free, &dt1);
          m_fileview_datatype_predef = false;
          stride = 1;
        }
        ++ip;
      }
      ++ms_axis;
    }
  }
  if (!m_fileview_datatype_predef)
    mpi_call(::MPI_Type_commit, &m_fileview_datatype);
}

void
Reader::adjust_outer_array_axis() {
  if (!m_inner_fileview_axis)
    return;
  MSColumns inner_fileview_axis = m_inner_fileview_axis.value();

  auto ip = std::begin(m_iter_params);
  while (ip != std::end(m_iter_params)
         && ip->axis != m_outer_array_axis
         && ip->axis != inner_fileview_axis)
    ++ip;

  while (ip != std::end(m_iter_params) && ip->axis != inner_fileview_axis) {
    auto next = ip + 1;
    if (next != std::end(m_iter_params)) {
      ip->in_array = false;
      m_outer_array_axis = next->axis;
    }
    ip = next;
  }
}

void
Reader::set_fileview(ArrayIndexer<MSColumns>::index& index) {
  // indices of inner array axes not required to be set by caller, set them to
  // zero
  bool fileview_axis = m_inner_fileview_axis.has_value();
  std::for_each(
    std::begin(m_iter_params),
    std::end(m_iter_params),
    [this, &fileview_axis, &index](const IterParams& ip) mutable {
      if (!fileview_axis)
        index[ip.axis] = 0;
      else if (ip.axis == m_inner_fileview_axis.value())
        fileview_axis = false;
    });

  std::size_t offset = m_ms_indexer->offset_of_(index);
  mpi_call(
    ::MPI_File_set_view,
    m_file,
    offset * sizeof(std::complex<float>),
    MPI_CXX_FLOAT_COMPLEX,
    m_fileview_datatype,
    "native",
    MPI_INFO_NULL);
}

std::vector<IndexBlockSequence<MSColumns> >
Reader::make_index_block_sequences() {
  std::vector<IndexBlockSequence<MSColumns> > result;
  bool in_array;
  std::for_each(
    std::begin(m_iter_params),
    std::end(m_iter_params),
    [this, &in_array, &result](const IterParams& ip) mutable {
      if (ip.axis == m_outer_array_axis)
        in_array = true;
      std::vector<IndexBlock> blocks;
      if (in_array) {
        // merge gap-less consecutive blocks
        std::size_t start = ip.origin;
        std::size_t end = ip.origin + ip.block_len;
        for (std::size_t b = 1; b < ip.max_blocks; ++b) {
          std::size_t s = ip.origin + b * ip.stride;
          if (s > end) {
            blocks.emplace_back(start, end - start);
            start = s;
          }
          end = s + ip.block_len;
        }
        blocks.emplace_back(start, end - start);
      } else {
        blocks.emplace_back(ip.origin, 1);
      }
      result.emplace_back(ip.axis, blocks);
    });
  return result;
}

std::shared_ptr<std::complex<float> >
Reader::read_array() {
  std::shared_ptr<std::complex<float> > result(
    reinterpret_cast<std::complex<float> *>(::operator new(m_buffer_size)));
  ::MPI_Status status;
  mpi_call(
    ::MPI_File_read_all,
    m_file,
    result.get(),
    1,
    m_array_datatype,
    &status);
  int count;
  mpi_call(::MPI_Get_count, &status, m_array_datatype, &count);
  if (count == 0)
    result.reset();
  return result;
}
