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

// CyclicGenerator
//
// generator for cyclic distribution of index blocks
//
class CyclicGenerator {

public:

  struct State {
    std::size_t axis_length;
    std::size_t block_length;
    std::size_t group_length;
    std::size_t index;
  };

  static std::tuple<
    State,
    std::optional<std::tuple<std::size_t, std::size_t> > >
  apply(const State& st) {

    auto index = st.index + st.group_length;
    if (index < st.axis_length) {
      State next_st {st.axis_length, st.block_length, st.group_length, index};
      auto blen = std::min(st.axis_length - index, st.block_length);
      return std::make_tuple(next_st, std::make_tuple(index, blen));
    } else {
      State next_st {
        st.axis_length, st.block_length, st.group_length, st.axis_length};
      return std::make_tuple(next_st, std::nullopt);
    }
  }
};

// BlockSequenceGenerator
//
// generator for sequence (vector) of blocks
//
class BlockSequenceGenerator {

public:

  struct State {
    std::vector<std::tuple<std::size_t, std::size_t> > blocks;
    std::size_t block_index;
  };

  static std::tuple<
    State,
    std::optional<std::tuple<std::size_t, std::size_t> > >
  apply(const State& st) {

    auto blk = st.block_index + 1;
    State next_st {st.blocks, blk};
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
      std::tuple<std::size_t, std::size_t> >::value>::type>
class BlockGenerator {

public:
  struct State {
    InputIterator current;
    InputIterator end;
  };

  static std::tuple<
    State,
    std::optional<std::tuple<std::size_t, std::size_t> > >
  apply(const State& st) {

    auto next(st.current);
    ++next;
    State next_st {next, st.end};
    if (next != st.end)
      return std::make_tuple(next_st, *next);
    else
      return std::make_tuple(next_st, std::nullopt);
  }
};

}

#endif // BLOCK_GENERATOR_H_
