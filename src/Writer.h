/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef WRITER_H_
#define WRITER_H_

#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <mpi.h>

#include <Reader.h>

namespace mpims {

template <typename T>
class Writer
  : public std::iterator<std::forward_iterator_tag, const MSArray<T>, std::size_t> {

public:
  Writer()
    : m_reader(Reader<T>()) {
  }

  Writer(Reader<T>&& reader)
    : m_reader(std::move(reader)) {
  }

  Writer(const Writer& other)
    : m_reader(other.m_reader)
    , m_array(other.m_array) {
  }

  Writer(Writer&& other)
    : m_reader(std::move(other).m_reader)
    , m_array(std::move(other).m_array) {
  }

  static const Writer
  end() {
    return Writer();
  };

  Writer&
  operator=(const Writer& other) {
    if (this != &other) {
      Writer temp(other);
      std::lock_guard<decltype(m_mtx)> lock(m_mtx);
      swap(temp);
    }
    return *this;
  }

  Writer&
  operator=(Writer&& other) {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    m_reader = std::move(other).m_reader;
    m_array = std::move(other).m_array;
    return *this;
  }

  Writer&
  operator++() {
    next();
    return *this;
  }

  Writer
  operator++(int) {
    Writer result(*this);
    operator++();
    return result;
  }

  bool
  operator==(const Writer& other) {
    if (other.at_end())
      return at_end();
    return (
      buffer_length() == other.buffer_length()
      && m_reader == other.m_reader
      && m_array == other.m_array);
  }

  bool
  operator!=(const Writer& other) {
    return !operator==(other);
  }

  bool
  at_end() const {
    return ((!m_reader.m_ms_shape
             || !(*m_reader.m_ms_shape)[0].is_indeterminate())
            && m_reader.at_end());
  };

  void
  next() {

    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    auto handles = m_reader.m_mpi_state.handles();
    std::lock_guard<decltype(*handles)> lck(*handles);
    if (handles->comm == MPI_COMM_NULL)
      return;

    if (m_reader.m_debug_log) {
      auto blocks = indices();
      std::ostringstream oss;
      oss << "(" << m_reader.m_rank << ") write ";
      const char* sep0 = "";
      std::for_each(
        std::begin(blocks),
        std::end(blocks),
        [&sep0, &oss](auto& ibs) {
          oss << sep0
              << mscol_nickname(ibs.m_axis) << ": [";
          sep0 = "; ";
          const char *sep1 = "";
          std::for_each(
            std::begin(ibs.m_blocks),
            std::end(ibs.m_blocks),
            [&sep1, &oss](auto& b) {
              oss << sep1 << "(" << b.m_index << "," << b.m_length << ")";
              sep1 = ",";
            });
          oss << "]";
        });
      oss << " at " << (m_array ? std::to_string(getv().offset()) : "--");
      oss << std::endl;
      std::clog << oss.str();
    }

    m_reader.extend();
    m_reader.set_deferred_fileview();
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
    m_array.reset();
    m_reader.next();
  }

  void
  interrupt() {
    m_reader.interrupt();
    m_reader = Reader<T>();
  }

  std::vector<IndexBlockSequence<MSColumns> >
  indices() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    return m_reader->blocks();
  };

  std::size_t
  buffer_length() const {
    return m_reader->num_elements() * sizeof(T);
  }

  std::optional<std::size_t>
  outer_min_index() const {
    const auto i = indices();
    if (i.size() > 0)
      return std::make_optional(i[0].min_index());
    else
      return std::nullopt;
  }

  void
  swap(Writer& other) {
    using std::swap;
    swap(m_reader, other.m_reader);
    swap(m_array, other.m_array);
  }

  unsigned
  num_ranks() const {
    return m_reader.num_ranks();
  }

  // NB: the returned array can be empty!
  const MSArray<T>&
  operator*() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    if (m_array)
      return m_array.value();
    else
      return getv();
  }

  MSArray<T>&
  operator*() {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    if (!m_array)
      m_array = MSArray<T>();
    return m_array.value();
  }

  MSArray<T>*
  operator->() {
    return &operator*();
  }

  static Writer
  begin(
    const std::string& path,
    const std::string& datarep,
    AMode access_mode,
    MPI_Comm comm,
    MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool debug_log = false) {

    std::vector<MSColumns> msao;
    std::transform(
      std::begin(ms_shape),
      std::end(ms_shape),
      std::back_inserter(msao),
      [](auto& ax) { return ax.id(); });

    return Writer(
      Reader<T>::wbegin(
        path,
        datarep,
        access_mode,
        comm,
        info,
        ms_shape,
        traversal_order,
        msao == traversal_order,
        pgrid,
        max_buffer_size,
        debug_log));
  }

protected:

  const MSArray<T>&
  getv() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    return *m_reader;
  };

  Reader<T> m_reader;

  std::optional<MSArray<T> > m_array;

  mutable std::recursive_mutex m_mtx;
};

template <typename T>
void
swap(Writer<T>& w1, Writer<T>& w2) {
  w1.swap(w2);
}

using CxFltWriter = Writer<std::complex<float> >;

using CxDblWriter = Writer<std::complex<double> >;

using FltWriter = Writer<float>;

using DblWriter = Writer<double>;

} // end namespace mpims

#endif // WRITER_H_
