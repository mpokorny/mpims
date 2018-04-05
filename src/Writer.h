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

class Writer
  : public std::iterator<std::forward_iterator_tag, const MSArray, std::size_t> {

public:
  Writer() {
  }

  Writer(Reader&& reader)
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

  static Writer
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
    return m_reader.at_end();
  };

  void
  next();

  void
  interrupt();

  std::vector<IndexBlockSequence<MSColumns> >
  indices() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    return m_reader.m_traversal_state.blocks();
  };

  std::size_t
  buffer_length() const {
    std::shared_ptr<const MPI_Datatype> dt;
    unsigned count;
    MPI_Count size;
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    if (m_reader.m_traversal_state.count == 0)
      return 0;
    std::tie(dt, count) = m_reader.m_traversal_state.buffer_datatype();
    MPI_Type_size_x(*dt, &size);
    assert(size % sizeof(std::complex<float>) == 0);
    return (size / sizeof(std::complex<float>)) * count;
  }

  void
  swap(Writer& other);

  unsigned
  num_ranks() const {
    return m_reader.num_ranks();
  }

  // NB: the returned array can be empty!
  const MSArray&
  operator*() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    if (m_array)
      return m_array.value();
    else
      return getv();
  }

  MSArray&
  operator*() {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    if (!m_array)
      m_array = MSArray();
    return m_array.value();
  }

  MSArray*
  operator->() {
    return &operator*();
  }

  static Writer
  begin(
    const std::string& path,
    const std::string& datarep,
    MPI_Comm comm,
    MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    bool debug_log = false);

protected:

  const MSArray&
  getv() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    return *m_reader;
  };

  Reader m_reader;

  std::optional<MSArray> m_array;

  mutable std::recursive_mutex m_mtx;
};

void
swap(Writer& w1, Writer& w2);

} // end namespace mpims

#endif // WRITER_H_
