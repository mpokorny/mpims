/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef INDEX_BLOCK_SEQUENCE_H_
#define INDEX_BLOCK_SEQUENCE_H_

#include <algorithm>
#include <vector>

namespace mpims {

struct IndexBlock {
	IndexBlock(std::size_t index, std::size_t length)
		: m_index(index)
		, m_length(length) {
	}

	std::size_t m_index;
	std::size_t m_length;

  bool
  operator==(const IndexBlock& other) const {
    return m_index == other.m_index && m_length == other.m_length;
  }

  bool
  operator!=(const IndexBlock& other) const {
    return !operator==(other);
  }
};

template <typename Axes>
struct IndexBlockSequence {
	IndexBlockSequence(Axes axis, const std::vector<IndexBlock>& blocks)
		: m_axis(axis)
		, m_blocks(blocks) {
	}

	IndexBlockSequence(Axes axis, std::vector<IndexBlock>&& blocks)
		: m_axis(axis)
		, m_blocks(std::move(blocks)) {
	}

	Axes m_axis;
	std::vector<IndexBlock> m_blocks;

  bool
  operator==(const IndexBlockSequence<Axes>& other) const {
    return m_axis == other.m_axis && m_blocks == other.m_blocks;
  }

  bool
  operator!=(const IndexBlockSequence<Axes>& other) const {
    return !operator==(other);
  }

  std::size_t
  num_elements() const {
    std::size_t result = 0;
    std::for_each(
      std::begin(m_blocks),
      std::end(m_blocks),
      [&result](const IndexBlock& ib) { result += ib.m_length; });
    return result;
  }
};

}; //

#endif // INDEX_BLOCK_H_
