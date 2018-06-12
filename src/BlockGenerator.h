#ifndef BLOCK_GENERATOR_H_
#define BLOCK_GENERATOR_H_

#include <cassert>
#include <functional>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace mpims {

typedef std::tuple<std::size_t, std::size_t> block_t;

// CyclicGenerator
//
// generator for cyclic distribution of index blocks
//
class CyclicGenerator {

public:

  struct State {
    std::size_t block_length;
    std::size_t group_length;
    std::size_t index;
    std::optional<std::size_t> axis_length;
  };

  static State
  initial_state(
    std::size_t block_length,
    std::size_t group_size,
    std::optional<std::size_t> axis_length,
    std::size_t group_index) {

    if (group_index >= group_size)
      throw std::domain_error("group_index is greater than or equal to group_size");

    std::size_t offset = group_index * block_length;

    return
      State{
        block_length,
        block_length * group_size,
        std::min(offset, axis_length.value_or(offset)),
        axis_length};
  }

  static std::tuple<State, std::optional<block_t> >
  apply(const State& st) {

    if (st.axis_length && st.index >= st.axis_length.value())
      return std::make_tuple(st, std::nullopt);

    auto blen = st.block_length;
    if (st.axis_length)
      blen = std::min(st.index + blen, st.axis_length.value()) - st.index;
    return
      std::make_tuple(
        State{
          st.block_length,
            st.group_length,
            st.index + st.group_length,
            st.axis_length},
        std::make_tuple(st.index, blen));
  }
};

// BlockSequenceGenerator
//
// generator for sequence (vector) of blocks
//
class BlockSequenceGenerator {

public:

  struct State {
    std::vector<block_t> blocks;
    std::optional<std::size_t> axis_length;
    std::size_t block_index;
    std::size_t block_offset;
  };

  static State
  initial_state(
    const std::vector<block_t>& blocks,
    std::optional<std::size_t> axis_length) {

    for (std::size_t i = 1; i < blocks.size(); ++i)
      if (std::get<0>(blocks[i])
          < std::get<0>(blocks[i - 1]) + std::get<1>(blocks[i - 1]))
        throw std::domain_error("overlapping blocks");

    auto brep =
      std::find_if(
        std::begin(blocks),
        std::end(blocks),
        [](auto& b) { return std::get<1>(b) == 0; });

    if (!axis_length && brep == std::end(blocks) && blocks.size() > 0) {
      std::size_t b0, blen;
      std::tie(b0, blen) = blocks[blocks.size() - 1];
      axis_length = b0 + blen;
    };

    if (brep != std::end(blocks))
      ++brep;

    return
      State{std::vector<block_t>(std::begin(blocks), brep), axis_length, 0, 0};  
  }

  static std::tuple<State, std::optional<block_t> >
  apply(const State& st) {

    if (st.block_index >= st.blocks.size())
      return std::make_tuple(st, std::nullopt);

    std::size_t b0, blen;
    std::tie(b0, blen) = st.blocks[st.block_index];
    b0 += st.block_offset;
    if (st.axis_length && b0 >= st.axis_length.value())
      return std::make_tuple(st, std::nullopt);

    if (st.axis_length)
      blen = std::min(b0 + blen, st.axis_length.value()) - b0;
    auto next_blk = st.block_index + 1;
    auto next_blk_offset = st.block_offset;
    std::size_t next_b0, next_blen;
    std::tie(next_b0, next_blen) = st.blocks[next_blk];
    if (next_blen == 0) {
      next_blk_offset += next_b0;
      if (!st.axis_length
          || next_blk_offset + std::get<0>(st.blocks[0]) < st.axis_length.value())
        next_blk = 0;
    }
    return std::make_tuple(
      State{st.blocks, st.axis_length, next_blk, next_blk_offset},
      std::make_tuple(b0, blen));
  }
};

}

#endif // BLOCK_GENERATOR_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
