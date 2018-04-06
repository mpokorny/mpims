/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef INDEX_BLOCK_SEQUENCE_H_
#define INDEX_BLOCK_SEQUENCE_H_

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>

namespace mpims {

struct IndexBlock {
	IndexBlock(std::size_t index, std::size_t length)
		: m_index(index)
		, m_length(length)
    , m_stride(std::nullopt) {
	}

	IndexBlock(std::size_t index, std::size_t length, std::size_t stride)
		: m_index(index)
		, m_length(length)
    , m_stride(stride) {
	}

	std::size_t m_index;
	std::size_t m_length;
  std::optional<std::size_t> m_stride; // present for unbounded sequences only

  bool
  operator==(const IndexBlock& other) const {
    return m_index == other.m_index
      && m_length == other.m_length
      && m_stride == other.m_stride;
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

  std::size_t
  min_index() const {
    return m_blocks[0].m_index;
  }

  std::size_t
  max_index() const {
    auto& last_block = m_blocks[m_blocks.size() - 1];
    return last_block.m_index + last_block.m_length - 1;
  }

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

  std::tuple<std::size_t, bool>
  num_elements() const {
    std::size_t ne = 0;
    bool unbounded = false;
    std::for_each(
      std::begin(m_blocks),
      std::end(m_blocks),
      [&ne,&unbounded](const IndexBlock& ib) {
        ne += ib.m_length;
        unbounded = unbounded || ib.m_stride;
      });
    return std::make_tuple(ne, unbounded);
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
    auto ibs = std::get<1>(*std::begin(m_sequences))[0];
    std::size_t stride = ibs.m_stride.value_or(index + 1);
    std::size_t offset = (index / stride) * stride;
    auto result = IndexBlockSequence(m_axis, m_sequences.at(index % stride));
    std::for_each(
      std::begin(result.m_blocks),
      std::end(result.m_blocks),
      [&offset](auto& blk) { blk.m_index += offset; });
    return result;
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
