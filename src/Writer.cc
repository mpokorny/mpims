/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <Writer.h>

using namespace mpims;

Writer
Writer::begin(
  const std::string& path,
  ::MPI_Comm comm,
  ::MPI_Info info,
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  bool ms_buffer_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t max_buffer_size,
  /* bool readahead, */
  bool debug_log) {

  return Writer(
    Reader::wbegin(
      path,
      comm,
      info,
      ms_shape,
      traversal_order,
      ms_buffer_order,
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
  if (m_buffer) {
    // a seek call isn't necessary when appending, but the current design of
    // Reader and Writer don't support that...in the future, could we just add a
    // call to !m_reader.at_end() as a condition on the seek when appending?
    mpi_call(
      ::MPI_File_seek,
      handles->file,
      -buffer_length(),
      MPI_SEEK_CUR);
    ::MPI_Status status;
    std::shared_ptr<const ::MPI_Datatype> dt;
    unsigned count;
    std::tie(dt, count) = m_reader.m_traversal_state.array_datatype();
    mpi_call(
      ::MPI_File_write_all,
      handles->file,
      m_buffer.get(),
      count,
      *dt,
      &status);
    // TODO: check status
  }
  m_reader.next();
}

void
Writer::interrupt() {
  m_reader.step(false);
}
