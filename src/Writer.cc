/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>

#include <Writer.h>

using namespace mpims;

Writer
Writer::begin(
  const std::string& path,
  ::MPI_Comm comm,
  ::MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t max_buffer_size,
  /* bool readahead, */
  bool debug_log) {

  std::vector<MSColumns> msao;
  std::transform(
    std::begin(ms_shape),
    std::end(ms_shape),
    std::back_inserter(msao),
    [](auto& ax) { return ax.id(); });

  return Writer(
    Reader::wbegin(
      path,
      comm,
      info,
      ms_shape,
      traversal_order,
      msao == traversal_order,
      pgrid,
      max_buffer_size,
      false /* readahead */,
      debug_log));
}

void
Writer::swap(Writer& other) {
  using std::swap;
  swap(m_reader, other.m_reader);
  swap(m_buffer, other.m_buffer);
}

void
Writer::next() {
  std::lock_guard<decltype(m_mtx)> lock(m_mtx);
  auto handles = m_reader.m_mpi_state.handles();
  std::lock_guard<decltype(*handles)> lck(*handles);
  if (m_reader.m_debug_log) {
    auto blocks = indices();
    std::clog << "(" << m_reader.m_rank << ") write ";
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
  if (m_buffer) {
    ::MPI_Offset offset = std::abs(std::get<2>(m_reader.m_ms_array));
    mpi_call(::MPI_File_seek, handles->file, offset, MPI_SEEK_SET);
  }
  ::MPI_Status status;
  std::shared_ptr<const ::MPI_Datatype> dt;
  unsigned count;
  std::tie(dt, count) = m_reader.m_traversal_state.buffer_datatype();
  if (!m_buffer)
    count = 0;
  mpi_call(
    ::MPI_File_write_all,
    handles->file,
    m_buffer.get(),
    count,
    *dt,
    &status);
  // TODO: check status
  m_buffer.reset();
  m_reader.next();
}

void
Writer::interrupt() {
  m_reader.step(false);
}

void
mpims::swap(Writer& w1, Writer& w2) {
  w1.swap(w2);
}
