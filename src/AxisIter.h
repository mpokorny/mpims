#ifndef AXIS_ITER_H_
#define AXIS_ITER_H_

#include <memory>

#include <mpims.h>
#include <IterParams.h>

namespace mpims {

// AxisIter contains the state of the iteration along an axis. It follows the
// iteration pattern defined for an axis by an IterParams value.
class AxisIter {
public:

  AxisIter(
    const std::shared_ptr<const IterParams>& params,
    bool outer_at_data)
    : m_params(params) {

    auto max_size = m_params->max_size();
    m_at_data = (!max_size || max_size.value() > 0) && outer_at_data;
    if (m_params->fully_in_array){
      m_take_n = m_params->size().value();
      m_at_buffer = false;
    } else if (m_params->buffer_capacity > 0) {
      m_take_n = m_params->buffer_capacity;
      m_at_buffer = true;
    } else {
      m_take_n = 1;
      m_at_buffer = false;
    }
    m_iter = m_params->begin();
    m_blocks = m_iter->take_blocked(m_take_n);
    m_index = 0;
  }

  AxisIter(const AxisIter& other)
    : m_params(other.m_params)
    , m_iter(m_params->begin())
    , m_take_n(other.m_take_n)
    , m_at_data(other.m_at_data)
    , m_at_buffer(other.m_at_buffer) {

    if (!other.at_end()) {
      m_blocks = m_iter->take_blocked(m_take_n);
      m_index = 0;
      while (m_index < other.m_index)
        take();
    } else {
      m_index = other.m_index;
    }
  }

  AxisIter(AxisIter&& other)
    : m_params(std::move(other).m_params)
    , m_iter(std::move(other).m_iter)
    , m_blocks(std::move(other).m_blocks)
    , m_take_n(other.m_take_n)
    , m_at_data(other.m_at_data)
    , m_at_buffer(other.m_at_buffer)
    , m_index(other.m_index) {
  }

  AxisIter&
  operator=(const AxisIter& rhs) {
    if (this != &rhs) {
      AxisIter temp(rhs);
      swap(temp);
    }
    return *this;
  }

  AxisIter&
  operator=(AxisIter&& rhs) {
    m_params = std::move(rhs).m_params;
    m_iter = std::move(rhs).m_iter;
    m_blocks = std::move(rhs).m_blocks;
    m_take_n = rhs.m_take_n;
    m_at_data = rhs.m_at_data;
    m_at_buffer = rhs.m_at_buffer;
    m_index = rhs.m_index;
    return *this;
  }

  std::vector<finite_block_t>
  take() {
    auto blocks = m_blocks;
    m_blocks = m_iter->take_blocked(m_take_n);
    ++m_index;
    if (m_at_data)
      return blocks;
    else
      return std::vector<finite_block_t>();
  }

  const std::vector<finite_block_t>&
  next_blocks() const {
    return m_blocks;
  }

  void
  complete() {
    while (m_blocks.size() > 0) {
      m_blocks = m_iter->take_blocked(m_take_n);
      ++m_index;
    }
  }

  bool
  at_data() const {
    return m_at_data;
  }

  bool
  at_end() const {
    return m_blocks.size() == 0;
  }

  MSColumns
  axis() const {
    return m_params->axis;
  }

  bool
  at_buffer() const {
    return m_at_buffer;
  }

  bool
  operator==(const AxisIter& rhs) const {
    return *m_params == *rhs.m_params && m_index == rhs.m_index;
  }

  bool
  operator!=(const AxisIter& rhs) const {
    return !operator==(rhs);
  }

protected:

  void
  swap(AxisIter& other) {
    using std::swap;
    swap(m_params, other.m_params);
    swap(m_iter, other.m_iter);
    swap(m_blocks, other.m_blocks);
    swap(m_take_n, other.m_take_n);
    swap(m_at_data, other.m_at_data);
    swap(m_at_buffer, other.m_at_buffer);
    swap(m_index, other.m_index);
  }

private:

  std::shared_ptr<const IterParams> m_params;
  std::unique_ptr<DataDistribution::Iterator> m_iter;
  std::vector<finite_block_t> m_blocks;
  std::size_t m_take_n;
  bool m_at_data;
  bool m_at_buffer;
  std::size_t m_index;
};

}

#endif // AXIS_ITER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
