/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>

#include <Writer.h>

using namespace mpims;

Writer
Writer::begin(
  const std::string& path,
  const std::string& datarep,
  MPI_Comm comm,
  MPI_Info info,
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
      datarep,
      comm,
      info,
      ms_shape,
      traversal_order,
      msao == traversal_order,
      pgrid,
      max_buffer_size,
      debug_log));
}

void
Writer::swap(Writer& other) {
  using std::swap;
  swap(m_reader, other.m_reader);
  swap(m_array, other.m_array);
}

void
Writer::next() {
  std::lock_guard<decltype(m_mtx)> lock(m_mtx);
  auto handles = m_reader.m_mpi_state.handles();
  std::lock_guard<decltype(*handles)> lck(*handles);
  if (handles->comm == MPI_COMM_NULL)
    return;
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
  if (m_array)
    MPI_File_seek(handles->file, getv().offset(), MPI_SEEK_SET);
  std::shared_ptr<const MPI_Datatype> dt;
  unsigned count;
  std::tie(dt, count) = m_reader.m_traversal_state.buffer_datatype();
  void *buff;
  if (!m_array || !m_array.value().buffer()) {
    count = 0;
    buff = nullptr;
  } else {
    buff = m_array.value().buffer().value();
  }
  MPI_Status status;
  MPI_File_write_all(handles->file, buff, count, *dt, &status);
  int st_count;
  MPI_Get_count(&status, *dt, &st_count);
  if (static_cast<unsigned>(st_count) != count)
    std::clog << "(" << m_reader.m_rank << ") "
              << "expected " << count
              << ", got " << st_count
              << std::endl;
  if (m_array)
    m_reader.m_ms_array.swap(m_array.value());
  m_array = std::nullopt;
  m_reader.next(false);
}

void
Writer::interrupt() {
  m_reader.interrupt();
}

void
mpims::swap(Writer& w1, Writer& w2) {
  w1.swap(w2);
}
