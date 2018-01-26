/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef INDEX_BLOCK_SEQUENCE_H_
#define INDEX_BLOCK_SEQUENCE_H_

#include <algorithm>
#include <unordered_map>
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

  void
  trim() {
    m_blocks.erase(
      std::remove_if(
        std::begin(m_blocks),
        std::end(m_blocks),
        [](auto& ib) { return ib.m_length == 0; }),
      std::end(m_blocks));
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

template <typename Axes>
struct IndexBlockSequenceMap {
  IndexBlockSequenceMap(
    Axes axis,
    const std::vector<std::vector<IndexBlock> >& blocks)
    : m_axis(axis) {
    std::for_each(
      std::begin(blocks),
      std::end(blocks),
      [this](const std::vector<IndexBlock>& blks) {
        m_sequences.emplace(blks[0].m_index, blks);
      });
  }

  Axes m_axis;
  std::unordered_map<std::size_t, std::vector<IndexBlock> > m_sequences;

  IndexBlockSequence<Axes>
  operator[](std::size_t index) const {
    return IndexBlockSequence(m_axis, m_sequences.at(index));
  }

  bool
  operator==(const IndexBlockSequenceMap& other) const {
    return m_axis == other.m_axis && m_sequences == other.m_sequences;
  }

  bool
  operator!=(const IndexBlockSequenceMap& other) const {
    return !operator==(other);
  }
};

}; //

#endif // INDEX_BLOCK_H_
