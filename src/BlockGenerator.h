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

// BlockGeneratorIterator
//
// index iterator from index block generator function
//
template <typename S>
class BlockGeneratorIterator
  : public std::iterator<std::input_iterator_tag, std::size_t> {

public:

  typedef std::function<
  std::tuple<
    S,
    std::optional<std::tuple<std::size_t, std::size_t> > >(const S&)>
  generator_t;

  BlockGeneratorIterator(
    const generator_t& generator,
    S&& state,
    bool at_end = false)
    : m_generator(generator)
    , m_block_offset(0) {
    if (!at_end)
      std::tie(m_next_state, m_block) = m_generator(state);
  }

  BlockGeneratorIterator(const BlockGeneratorIterator& other)
    : m_generator(other.m_generator)
    , m_next_state(other.m_next_state)
    , m_block(other.m_block)
    , m_block_offset(other.m_block_offset) {
  }

  bool
  operator==(const BlockGeneratorIterator& rhs) {
    if (at_end() && rhs.at_end())
      return true;
    if (at_end() || rhs.at_end())
      return false;
    return (m_block == rhs.m_block
            && m_block_offset == rhs.m_block_offset
            && m_generator == rhs.m_generator);
  }

  bool
  operator!=(const BlockGeneratorIterator& rhs) {
    return !operator==(rhs);
  }

  std::size_t
  operator*() const {
    std::size_t b0;
    std::tie(b0, std::ignore) = m_block.value();
    return b0 + m_block_offset;
  }

  BlockGeneratorIterator&
  operator++() {
    std::size_t b0, blen;
    std::tie(b0, blen) = m_block.value();
    if (++m_block_offset == blen) {
      std::tie(m_next_state, m_block) = m_generator(m_next_state);
      assert(!m_block || std::get<0>(m_block.value()) >= b0 + blen);
      m_block_offset = 0;
    }
    return *this;
  }

  BlockGeneratorIterator
  operator++(int) {
    BlockGeneratorIterator result(*this);
    operator++();
    return result;
  }

  bool
  at_end() const {
    return !m_block;
  }

  std::vector<std::size_t>
  take(std::size_t n = 1) {
    std::vector<std::size_t> result;
    result.resize(n);
    while (n > 0 && !at_end()) {
      result.push_back(operator*());
      operator++();
    }
    return result;
  }

protected:

  void
  swap(BlockGeneratorIterator& other) {
    using std::swap;
    swap(m_generator, other.m_generator);
    swap(m_next_state, other.m_next_state);
    swap(m_block, other.m_block);
    swap(m_block_offset, other.m_block_offset);
  }

private:

  generator_t m_generator;
  S m_next_state;
  std::optional<std::tuple<std::size_t, std::size_t> > m_block;
  std::size_t m_block_offset;
};

}

#endif // BLOCK_GENERATOR_H_
