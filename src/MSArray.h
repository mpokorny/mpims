#ifndef MS_ARRAY_H_
#define MS_ARRAY_H_

#include <algorithm>
#include <complex>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <vector>

#include <mpi.h>

#include <IndexBlock.h>
#include <MSColumns.h>

namespace mpims {

template <typename T>
struct MSArray {

  MSArray()
    : m_count(0)
    , m_req_or_st(std::in_place_index<1>) {
  }

  MSArray(std::size_t len)
    : m_buffer(reinterpret_cast<T *>(::operator new(len * sizeof(T))))
    , m_count(0)
    , m_req_or_st(std::in_place_index<1>) {
  }

  MSArray(
    std::vector<IndexBlockSequence<MSColumns> >&& blocks,
    std::optional<int> rank=std::nullopt)
    : m_blocks(std::move(blocks))
    , m_count(0)
    , m_req_or_st(std::in_place_index<1>)
    , m_rank(rank) {
  }

  MSArray(
    std::vector<IndexBlockSequence<MSColumns> >&& blocks,
    std::unique_ptr<T>&& buffer,
    std::optional<MPI_Offset>&& offset,
    int count,
    std::shared_ptr<const MPI_Datatype>& datatype,
    MPI_Request&& request,
    std::optional<int> rank=std::nullopt)
    : m_blocks(std::move(blocks))
    , m_buffer(std::move(buffer))
    , m_offset(std::move(offset))
    , m_count(count)
    , m_datatype(datatype)
    , m_req_or_st(std::move(request))
    , m_rank(rank) {
  }

  MSArray(
    std::vector<IndexBlockSequence<MSColumns> >&& blocks,
    std::unique_ptr<T>&& buffer,
    std::optional<MPI_Offset>&& offset,
    int count,
    std::shared_ptr<const MPI_Datatype>& datatype,
    MPI_Status&& status,
    std::optional<int> rank=std::nullopt)
    : m_blocks(std::move(blocks))
    , m_buffer(std::move(buffer))
    , m_offset(std::move(offset))
    , m_count(count)
    , m_datatype(datatype)
    , m_req_or_st(std::move(status))
    , m_rank(rank) {
  }

  MSArray(const MSArray& other)
    : m_blocks(other.m_blocks)
    , m_offset(other.m_offset)
    , m_count(other.m_count)
    , m_datatype(other.m_datatype)
    , m_rank(other.m_rank) {
    other.wait();
    std::size_t sz = other.num_elements() * sizeof(T);
    if (sz > 0 && other.m_buffer) {
      m_buffer.reset(reinterpret_cast<T *>(::operator new(sz)));
      memcpy(m_buffer.get(), other.m_buffer.get(), sz);
    }
    m_req_or_st = other.m_req_or_st;
  }

  MSArray(MSArray&& other)
    : m_blocks(std::move(other).m_blocks)
    , m_buffer(std::move(other).m_buffer)
    , m_offset(std::move(other).m_offset)
    , m_count(std::move(other).m_count)
    , m_datatype(std::move(other).m_datatype)
    , m_req_or_st(std::move(other).m_req_or_st)
    , m_rank(std::move(other).m_rank) {
  }

  MSArray&
  operator=(const MSArray& other) {
    if (this != &other) {
      MSArray temp(other);
      swap(temp);
    }
    return *this;
  }

  MSArray&
  operator=(MSArray&& other) {
    m_blocks = std::move(other).m_blocks;
    m_buffer = std::move(other).m_buffer;
    m_offset = std::move(other).m_offset;
    m_count = std::move(other).m_count;
    m_datatype = std::move(other).m_datatype;
    std::swap(m_req_or_st, other.m_req_or_st);
    m_rank = std::move(other).m_rank;
    return *this;
  }

  bool
  operator==(const MSArray& other) const {
    return (
      m_blocks == other.m_blocks
      && m_count == other.m_count
      && m_offset == other.m_offset
      && *m_datatype == *other.m_datatype
      && ((!m_buffer && !other.m_buffer)
          || (m_buffer && other.m_buffer
              && std::memcmp(
                m_buffer.get(),
                other.m_buffer.get(),
                num_elements() * sizeof(*m_buffer)))));
  }

  bool
  operator!=(const MSArray& other) const {
    return !operator==(other);
  }

  size_t
  num_elements() const {
    size_t result = (m_blocks.size() > 0) ? 1 : 0;
    std::for_each(
      std::begin(m_blocks),
      std::end(m_blocks),
      [&result](const IndexBlockSequence<MSColumns>& ibs) {
        result *= std::get<0>(ibs.num_elements());
      });
    return result;
  }

  void
  wait(bool cancel=false) const {

    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    if (std::holds_alternative<MPI_Request>(m_req_or_st)) {
      MPI_Request& req = std::get<MPI_Request>(m_req_or_st);
      if (cancel)
        MPI_Cancel(&req);
      MPI_Status st;
      MPI_Wait(&req, &st);
      m_req_or_st = std::move(st);
      test_status();
    }
  }

  const std::vector<IndexBlockSequence<MSColumns> >&
  blocks() const {
    return m_blocks;
  }

  std::optional<T *>
  buffer() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    wait();
    return m_buffer ? std::make_optional(m_buffer.get()) : std::nullopt;
  };

  std::optional<MPI_Offset>
  offset() const {
    return m_offset;
  }

  void
  test_status() const {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    wait();
    int st_count;
    MPI_Get_count(&std::get<MPI_Status>(m_req_or_st), *m_datatype, &st_count);
    if (m_count != st_count) {
      m_buffer.reset();
      if (m_rank)
        std::clog << "(" << m_rank.value() << ") "
                  << "expected " << m_count
                  << ", got " << st_count
                  << std::endl;
    }
  }

  void
  swap(MSArray& other) {
    std::lock_guard<decltype(m_mtx)> lock(m_mtx);
    std::lock_guard<decltype(m_mtx)> lock1(other.m_mtx);
    using std::swap;
    swap(m_blocks, other.m_blocks);
    swap(m_buffer, other.m_buffer);
    swap(m_offset, other.m_offset);
    swap(m_count, other.m_count);
    swap(m_datatype, other.m_datatype);
    swap(m_req_or_st, other.m_req_or_st);
    swap(m_rank, other.m_rank);
  }

private:

  mutable std::recursive_mutex m_mtx;

  std::vector<IndexBlockSequence<MSColumns> > m_blocks;

  mutable std::unique_ptr<T> m_buffer;

  std::optional<MPI_Offset> m_offset;

  int m_count;

  std::shared_ptr<const MPI_Datatype> m_datatype;

  mutable std::variant<MPI_Request, MPI_Status> m_req_or_st;

  std::optional<int> m_rank;
};

template <typename T>
void
swap(MSArray<T>& array1, MSArray<T>& array2) {
  array1.swap(array2);
}

using CxFltMSArray = MSArray<std::complex<float> >;

using CxDblMSArray = MSArray<std::complex<double> >;

using FltMSArray = MSArray<float>;

using DblMSArray = MSArray<double>;

}

#endif // MS_ARRAY_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
