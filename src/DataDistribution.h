/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef DATA_DISTRIBUTION_H_
#define DATA_DISTRIBUTION_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>
#include <tuple>
#include <vector>

namespace mpims {

// blocks()
//
// create vector of blocks from iterator over indices
//
template <
  typename InputIterator,
  class = typename std::enable_if<
    std::is_convertible<
      typename std::iterator_traits<InputIterator>::value_type,
      std::size_t>::value>::type>
std::vector<std::tuple<std::size_t, std::size_t> >
blocks(const InputIterator& first, const InputIterator& last) {
  std::vector<std::tuple<std::size_t, std::size_t> > result;
  std::optional<std::tuple<std::size_t, std::size_t> > block;
  std::for_each(
    first,
    last,
    [&result, &block](const auto& i) {
      if (block) {
        auto blk = block.value();
        std::size_t b0 = 0, blen = 0;
        std::tie(b0, blen) = blk;
        auto bend = b0 + blen;
        if (i > bend) {
          result.push_back(blk);
          block.emplace(i, 1);
        } else {
          assert(i == bend);
          block.emplace(b0, blen + 1);
        }
      } else {
        block.emplace(i, 1);
      }
    });
  if (block)
    result.push_back(block.value());
  return result;
}

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

// CyclicGenerator
//
// generator function for cyclic distribution of index blocks
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

// CyclicIterator
//
// index iterator for block-cyclic distribution
//
class CyclicIterator
  : public BlockGeneratorIterator<CyclicGenerator::State> {

public:

  CyclicIterator(
    std::size_t axis_length,
    std::size_t offset,
    std::size_t block_length,
    std::size_t group_size,
    bool at_end = false)
    : BlockGeneratorIterator(
      CyclicGenerator::apply,
      CyclicGenerator::State{
        axis_length,
          block_length,
          block_length * group_size,
          at_end ? axis_length : offset},
      at_end) {
  }
};

// BlockSequenceGenerator
//
// generator function for sequence (vector) of blocks
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

// BlockSequenceIterator
//
// index iterator for sequence of blocks
//
class BlockSequenceIterator
  : public BlockGeneratorIterator<BlockSequenceGenerator::State> {

public:

  BlockSequenceIterator(
    const std::vector<std::tuple<std::size_t, std::size_t> >& blocks,
    bool at_end = false)
    : BlockGeneratorIterator(
      BlockSequenceGenerator::apply,
      BlockSequenceGenerator::State {blocks, at_end ? blocks.size() : 0},
      at_end) {
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

// BlockIterator
//
// index iterator for block-yielding iterator
//
template <
  typename InputIterator,
  class = typename std::enable_if<
    std::is_convertible<
      typename std::iterator_traits<InputIterator>::value_type,
      std::tuple<std::size_t, std::size_t> >::value>::type>
class BlockIterator
  : public BlockGeneratorIterator<typename BlockGenerator<InputIterator>::State> {

public:

  BlockIterator(InputIterator&& begin, InputIterator&& end)
    : BlockGeneratorIterator<BlockGenerator<InputIterator> >(
      BlockGenerator<InputIterator>::apply,
      typename BlockGenerator<InputIterator>::State {begin, end},
      begin == end) {
  }
};

// DataDistribution
//
// abstract class for data access distributions
//
// use DataDistributionFactory to create instances
//
class DataDistribution {
public:
  class Iterator
    : public std::iterator<std::input_iterator_tag, std::size_t> {

  public:

    virtual std::size_t
    operator*() const = 0;

    virtual Iterator&
    operator++() = 0;

    virtual bool
    at_end() const = 0;

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

    std::vector<std::tuple<std::size_t, std::size_t> >
    take_blocked(std::size_t n = 1) {
      auto seq = take(n);
      return mpims::blocks(std::begin(seq), std::end(seq));
    }

    std::vector<std::size_t>
    take_all() {
      std::vector<std::size_t> result;
      while (!at_end()) {
        result.push_back(operator*());
        operator++();
      }
      return result;
    }

    std::vector<std::tuple<std::size_t, std::size_t> >
    take_all_blocked() {
      auto seq = take_all();
      return mpims::blocks(std::begin(seq), std::end(seq));
    }
  };

  virtual std::unique_ptr<Iterator>
  begin() const = 0;

  std::vector<std::tuple<std::size_t, std::size_t> >
  blocks() const {
    return begin()->take_all_blocked();
  }

  std::string
  show() const {
    std::ostringstream result("[");
    const char *sep = "";
    for (const auto& blk : begin()->take_all_blocked()) {
      result << sep
             << "(" << std::get<0>(blk) << "," << std::get<1>(blk) << ")";
      sep = ",";
    }
    result << "]";
    return result.str();
  }
};

// IteratorDataDistribution
//
// data distribution for index-valued iterator
//
// base class for all sub-classes of DataDistribution defined in this module
//
template <
  typename T, 
  class = typename std::enable_if<
    std::is_convertible<
      typename std::iterator_traits<T>::value_type,
      std::size_t>::value>::type>
class IteratorDataDistribution
  : public DataDistribution {

public:

  typedef T iterator;

  IteratorDataDistribution(iterator&& begin)
    : m_begin(std::forward<iterator>(begin)) {
  }

  IteratorDataDistribution(const IteratorDataDistribution& other)
    : m_begin(other.m_begin) {
  }

  IteratorDataDistribution(IteratorDataDistribution&& other)
    : m_begin(std::move(other).m_begin) {
  }

  IteratorDataDistribution&
  operator=(const IteratorDataDistribution& other) {
    if (this != &other) {
      IteratorDataDistribution temp(other);
      swap(temp);
    }
    return *this;
  }

  IteratorDataDistribution&
  operator=(IteratorDataDistribution&& other) {
    m_begin = std::move(other).m_begin;
    return *this;
  }

  class IteratorDataDistributionIterator
    : public Iterator {

  public:

    IteratorDataDistributionIterator(const T& iter)
      : m_iter(iter) {
    }

    std::size_t
    operator*() const override {
      return *m_iter;
    }

    Iterator&
    operator++() override {
      ++m_iter;
      return *this;
    }

    bool
    at_end() const override {
      return m_iter.at_end();
    }

  private:
    T m_iter;
  };

  std::unique_ptr<Iterator>
  begin() const override {
    return std::make_unique<IteratorDataDistributionIterator>(m_begin);
  }

  template <typename U>
  bool
  operator==(const IteratorDataDistribution<U>& other) {
    iterator i = itbegin();
    U j = other.itbegin();
    if (i.at_end() && j.at_end())
      return true;
    bool eq = true;
    while (!i.at_end() && !j.at_end() && eq) {
      eq = *i == *j;
      ++i;
      ++j;
    }
    return eq && i.at_end() && j.at_end();
  }

  template <typename U>
  bool
  operator!=(const IteratorDataDistribution<U>& other) {
    return !operator==(other);
  }

protected:

  iterator
  itbegin() const {
    return m_begin;
  }

  void
  swap(IteratorDataDistribution& other) {
    m_begin.swap(other.m_begin);
    
  }

private:

  iterator m_begin;
};

// BlockGeneratorDataDistribution
//
// data distribution for a BlockGeneratorIterator type
//
// various sub-types of BlockGeneratorIterator defined in this module also have
// direct sub-classes of IteratorDataDistribution defined here, but this
// provides a generic definition without any sort of specialized name or
// constructors
//
template <typename S>
class BlockGeneratorDataDistribution
  : public IteratorDataDistribution<BlockGeneratorIterator<S> > {

public:

  BlockGeneratorDataDistribution(
    const typename BlockGeneratorIterator<S>::generator_t& generator,
    const S& state)
    : IteratorDataDistribution<BlockGeneratorIterator<S> >(
      BlockGeneratorIterator(generator, state)) {
  }
};

// CyclicDataDistribution
//
// block-cyclic data distribution
//
class CyclicDataDistribution
  : public IteratorDataDistribution<CyclicIterator> {

public:

  CyclicDataDistribution(
    std::size_t axis_length,
    std::size_t offset,
    std::size_t block_length,
    std::size_t group_size)
    : IteratorDataDistribution(
      CyclicIterator(
        axis_length,
        offset,
        block_length,
        group_size)) {
  }
};

// BlockSequenceDataDistribution
//
// (enumerated) block sequence data distribution
//
class BlockSequenceDataDistribution
  : public IteratorDataDistribution<BlockSequenceIterator> {

public:

  BlockSequenceDataDistribution(
    const std::vector<std::tuple<std::size_t, std::size_t> >& blocks)
    : IteratorDataDistribution(BlockSequenceIterator(blocks)) {
  }
};

// BlockIteratorDataDistribution
//
// block iterator data distribution
//
template <
  typename InputIterator,
  class = typename std::enable_if<
    std::is_convertible<
      typename std::iterator_traits<InputIterator>::value_type,
      std::tuple<std::size_t, std::size_t> >::value>::type>
class BlockIteratorDataDistribution
  : public IteratorDataDistribution<BlockIterator<InputIterator> > {

public:

  BlockIteratorDataDistribution(InputIterator&& begin, InputIterator&& end)
    : IteratorDataDistribution<BlockIterator<InputIterator> >(
      BlockIterator(
        std::forward<InputIterator>(begin),
        std::forward<InputIterator>(end))) {
  }
};

class DataDistributionFactory {

public:

  static std::unique_ptr<DataDistribution>
  cyclic(
    std::size_t axis_length,
    std::size_t offset,
    std::size_t block_length,
    std::size_t group_size) {

  return
    std::make_unique<CyclicDataDistribution>(
      axis_length,
      offset,
      block_length,
      group_size);
  }

  static std::unique_ptr<DataDistribution>
  block_sequence(
    const std::vector<std::tuple<std::size_t, std::size_t> >& blocks) {

    return std::make_unique<BlockSequenceDataDistribution>(blocks);
  }
  
  template <
    typename InputIterator,
    class = typename std::enable_if<
      std::is_convertible<
        typename std::iterator_traits<InputIterator>::value_type,
        std::tuple<std::size_t, std::size_t> >::value>::type>
  static std::unique_ptr<DataDistribution>
  block_iterator(InputIterator&& begin, InputIterator&& end) {

    return
      std::make_unique<BlockIteratorDataDistribution>(
        std::forward<InputIterator>(begin),
        std::forward<InputIterator>(end));
  }

  template <typename S>
  static std::unique_ptr<DataDistribution>
  block_generator(
    const typename BlockGeneratorIterator<S>::generator_t& generator,
    const S& init_state) {

    return
      std::make_unique<BlockGeneratorIterator<S> >(
        generator,
        init_state);
  }
};

}

#endif // DATA_DISTRIBUTION_H_
