/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef COLUMN_AXIS_H_
#define COLUMN_AXIS_H_

#include <string>

namespace mpims {

template <typename Columns>
class ColumnAxisBase {
public:

  Columns
  id() const {
    return static_cast<Columns>(m_col);
  }

  std::size_t
  length() const {
    return m_length;
  }

  ColumnAxisBase(unsigned col, std::size_t length)
    : m_col(col)
    , m_length(length) {
  }

protected:
  unsigned m_col;

  std::size_t m_length;
};

template <typename Columns,
          Columns Column>
class ColumnAxis
  : public ColumnAxisBase<Columns> {

public:

  ColumnAxis(std::size_t length)
    : ColumnAxisBase<Columns>(static_cast<unsigned>(Column), length) {
  }

  ColumnAxis(const ColumnAxis& other)
    : ColumnAxisBase<Columns>(other.m_col, other.m_length) {
  }

  ColumnAxis(ColumnAxis&& other)
    : ColumnAxisBase<Columns>(std::forward<ColumnAxis>(other).m_col,
                              std::forward<ColumnAxis>(other).m_length) {
  }

  ColumnAxis&
  operator=(ColumnAxis other) = delete;

  ColumnAxis&
  operator=(ColumnAxis&& other) = delete;

  ColumnAxis
  set_length(std::size_t length) const {
    return ColumnAxis(length);
  }
};

} // end namespace mpims

#endif // COLUMN_AXIS_H_
