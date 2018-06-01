/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef BLOCK_GENERATOR_H_
#define BLOCK_GENERATOR_H_

#include <cassert>
#include <functional>
#include <iterator>
#include <optional>
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

  static std::tuple<State, std::optional<block_t> >
  apply(const State& st) {

    if (st.block_index >= st.blocks.size())
      return std::make_tuple(
        State{st.blocks, st.axis_length, st.blocks.size(), st.block_offset},
        std::nullopt);

    auto blk_offset = st.block_offset;
    std::size_t b0, blen;
    std::tie(b0, blen) = st.blocks[blk];
    if (blen == 0) {
      blk_offset += b0;
      blk = 0;
      std::tie(b0, blen) = st.blocks[blk];
      b0 += blk_offset;
    }
    State next_st {st.blocks, st.axis_length, blk, blk_offset};
    if (blk < st.blocks.size())
      return std::make_tuple(next_st, st.blocks[blk]);
    else
      return std::make_tuple(next_st, std::nullopt);
  }
};

// BlockGenerator
//
// generator function for block-valued iterator
//
template <
  typename InputIterator,
  class = typename std::enable_if<
    std::is_convertible<
      typename std::iterator_traits<InputIterator>::value_type,
      block_t>::value>::type>
class BlockGenerator {

public:
  struct State {
    InputIterator current;
    InputIterator end;
  };

  static std::tuple<State, std::optional<block_t> >
  apply(const State& st) {

    if (st.current == st.end)
      return std::make_tuple(st, std::nullopt);

    auto next(st.current);
    ++next;
    if (next != st.end)
      return std::make_tuple(State{next, st.end}, *next);
    else
      return std::make_tuple(State{st.end, st.end}, std::nullopt);
  }
};

}

#endif // BLOCK_GENERATOR_H_
