/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef INDEX_BLOCK_SEQUENCE_H_
#define INDEX_BLOCK_SEQUENCE_H_

#include <vector>

namespace mpims {

struct IndexBlock {
	IndexBlock(std::size_t index, std::size_t length)
		: m_index(index)
		, m_length(length) {
	}

	std::size_t m_index;
	std::size_t m_length;
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
};

}; //

#endif // INDEX_BLOCK_H_
