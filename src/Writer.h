/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef WRITER_H_
#define WRITER_H_

#include <cstring>
#include <memory>
#include <vector>

#include <mpi.h>

#include <Reader.h>

namespace mpims {

class Writer {

public:
  Writer() {
  }

  Writer(Reader&& reader)
    : m_reader(std::move(reader)) {
  }

  Writer(const Writer& other)
    : m_reader(other.m_reader) {

    if (m_buffer) {
      std::size_t buffer_size = buffer_length() * sizeof(std::complex<float>);
      std::unique_ptr<std::complex<float> > new_buffer(
        reinterpret_cast<std::complex<float> *>(::operator new(buffer_size)));
      std::lock_guard<decltype(m_mtx)> lock(other.m_mtx);
      memcpy(new_buffer.get(), other.m_buffer.get(), buffer_size);
      m_buffer = std::move(new_buffer);
    }
  }

  Writer(Writer&& other)
    : m_reader(std::move(other).m_reader)
    , m_buffer(std::move(other).m_buffer) {
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
    m_buffer = std::move(other).m_buffer;
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
      && memcmp(
        m_buffer.get(),
        other.m_buffer.get(),
        sizeof(std::complex<float>) * buffer_length()) == 0);
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
    std::shared_ptr<const ::MPI_Datatype> dt;
    unsigned count;
    ::MPI_Count size;
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    if (m_reader.m_traversal_state.count == 0)
      return 0;
    std::tie(dt, count) = m_reader.m_traversal_state.buffer_datatype();
    mpi_call(::MPI_Type_size_x, *dt, &size);
    assert(size % sizeof(std::complex<float>) == 0);
    return (size / sizeof(std::complex<float>)) * count;
  }

  // this is purely a convenience method, it's not necessary for a client to use
  // it to allocate a write buffer
  std::unique_ptr<std::complex<float> >
  allocate_buffer() const {
    return
      std::unique_ptr<std::complex<float> >(
        reinterpret_cast<std::complex<float> *>(
          ::operator new(sizeof(std::complex<float>) * buffer_length())));
  }

  void
  swap(Writer& other);

  unsigned
  num_ranks() const {
    return m_reader.num_ranks();
  }

  // NB: the returned array can be empty! Caller should test for null.
  const std::shared_ptr<const std::complex<float> >
  operator*() const {
    return getv();
  }

  std::unique_ptr<const std::complex<float> >&
  operator*() {
    return m_buffer;
  }

  std::unique_ptr<std::shared_ptr<const std::complex<float> > >
  operator->() const {
    std::unique_ptr<std::shared_ptr<const std::complex<float> > > result;
    *result = getv();
    return result;
  }

  static Writer
  begin(
    const std::string& path,
    const std::string& datarep,
    ::MPI_Comm comm,
    ::MPI_Info info,
    const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
    const std::vector<MSColumns>& traversal_order,
    std::unordered_map<MSColumns, DataDistribution>& pgrid,
    std::size_t max_buffer_size,
    /* bool readahead, */
    bool debug_log = false);

protected:

  const std::shared_ptr<const std::complex<float> >
  getv() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    return *m_reader;
  };

  Reader m_reader;

  std::unique_ptr<const std::complex<float> > m_buffer;

  mutable std::recursive_mutex m_mtx;
};

void
swap(Writer& w1, Writer& w2);

} // end namespace mpims

#endif // WRITER_H_
