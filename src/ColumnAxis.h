#ifndef COLUMN_AXIS_H_
#define COLUMN_AXIS_H_

#include <optional>
#include <string>

namespace mpims {

template <typename Columns>
class ColumnAxisBase {
public:

  ColumnAxisBase(unsigned col, std::size_t length)
    : m_col(col)
    , m_length(length) {
  }

  ColumnAxisBase(unsigned col)
    : m_col(col) {
  }

  Columns
  id() const {
    return static_cast<Columns>(m_col);
  }

  std::optional<std::size_t>
  length() const {
    return m_length;
  }

  bool
  is_indeterminate() const {
    return !m_length;
  }

protected:
  unsigned m_col;

  std::optional<std::size_t> m_length;
};

template <typename Columns,
          Columns Column>
class ColumnAxis
  : public ColumnAxisBase<Columns> {

public:

  ColumnAxis()
    : ColumnAxisBase<Columns>(static_cast<unsigned>(Column)) {
  }

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

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
