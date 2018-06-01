/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef DATA_DISTRIBUTION_H_
#define DATA_DISTRIBUTION_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>
#include <tuple>
#include <vector>

#include <BlockGenerator.h>

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

class DataDistributionFactory;

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

// GeneratorDataDistribution
//
// data distribution for block-valued generator
//
// this is the implementation sub-class of DataDistribution, which should
// support almost all sorts of data distributions (all that's required is a
// generator function and an initial state)
//
template <typename S>
class GeneratorDataDistribution
  : public DataDistribution {

public:

  typedef std::function<
  std::tuple<
    S,
    std::optional<std::tuple<std::size_t, std::size_t> > >(const S&)>
  generator_t;

  friend std::unique_ptr<GeneratorDataDistribution>
  std::make_unique<GeneratorDataDistribution>(const generator_t&, S&&);

  static std::unique_ptr<DataDistribution>
  make(const generator_t& generator, S&& state) {
    return
      std::make_unique<GeneratorDataDistribution>(
        generator,
        std::forward<S>(state));
  }

  GeneratorDataDistribution&
  operator=(const GeneratorDataDistribution& other) {
    if (this != &other) {
      GeneratorDataDistribution temp(other);
      swap(temp);
    }
    return *this;
  }

  GeneratorDataDistribution&
  operator=(GeneratorDataDistribution&& other) {
    m_generator = std::move(other).m_generator;
    m_init_state = std::move(other).m_init_state;
    return *this;
  }

  class GeneratorIterator
    : public Iterator {

    friend std::unique_ptr<GeneratorIterator>
    std::make_unique<GeneratorIterator>(const generator_t&, const S&);

  public:

    std::size_t
    operator*() const override {
      std::size_t b0;
      std::tie(b0, std::ignore) = m_block.value();
      return b0 + m_block_offset;
    }

    Iterator&
    operator++() override {
      std::size_t b0, blen;
      std::tie(b0, blen) = m_block.value();
      if (++m_block_offset == blen) {
        std::tie(m_next_state, m_block) = m_generator(m_next_state);
        assert(!m_block || std::get<0>(m_block.value()) >= b0 + blen);
        m_block_offset = 0;
      }
      return *this;
    }

    bool
    at_end() const override {
      return !m_block;
    }

  protected:

    GeneratorIterator(const generator_t& generator, const S& state)
      : m_generator(generator)
      , m_block_offset(0) {

      std::tie(m_next_state, m_block) = m_generator(state);
    }

  private:

    generator_t m_generator;
    S m_next_state;
    std::optional<std::tuple<std::size_t, std::size_t> > m_block;
    std::size_t m_block_offset;
  };

  std::unique_ptr<Iterator>
  begin() const override {
    return std::make_unique<GeneratorIterator>(m_generator, m_init_state);
  }

protected:

  void
  swap(GeneratorDataDistribution& other) {
    using std::swap;
    swap(m_generator, other.m_generator);
    swap(m_init_state, other.m_init_state);
  }

private:

  GeneratorDataDistribution(const generator_t& generator, const S& state)
    : m_generator(generator)
    , m_init_state(state) {
  }

  GeneratorDataDistribution(const GeneratorDataDistribution& other)
    : m_generator(other.m_generator)
    , m_init_state(other.m_init_state) {
  }

  GeneratorDataDistribution(GeneratorDataDistribution&& other)
    : m_generator(std::move(other).m_generator)
    , m_init_state(std::move(other).m_init_state) {
  }

  generator_t m_generator;

  S m_init_state;
};

class DataDistributionFactory {

public:

  // block-cyclic data distribution
  //
  static std::unique_ptr<DataDistribution>
  cyclic(
    std::size_t axis_length,
    std::size_t offset,
    std::size_t block_length,
    std::size_t group_size) {

    return
      GeneratorDataDistribution<CyclicGenerator::State>::make(
        CyclicGenerator::apply,
        CyclicGenerator::State{
          axis_length,
            block_length,
            block_length * group_size,
            std::min(offset, axis_length)});
  }

  // (enumerated) block sequence data distribution
  //
  static std::unique_ptr<DataDistribution>
  block_sequence(
    const std::vector<std::tuple<std::size_t, std::size_t> >& blocks) {

    return
      GeneratorDataDistribution<BlockSequenceGenerator::State>::make(
        BlockSequenceGenerator::apply,
        BlockSequenceGenerator::State{blocks, 0});
  }

  // block iterator data distribution
  //
  template <
    typename InputIterator,
    class = typename std::enable_if<
      std::is_convertible<
        typename std::iterator_traits<InputIterator>::value_type,
        std::tuple<std::size_t, std::size_t> >::value>::type>
  static std::unique_ptr<DataDistribution>
  block_iterator(InputIterator&& begin, InputIterator&& end) {

    return
      GeneratorDataDistribution<
        typename BlockGenerator<InputIterator>::State>::make(
          BlockGenerator<InputIterator>::apply,
          typename BlockGenerator<InputIterator>::State{begin, end});
  }

  // generic data distribution factory method
  //
  template <typename S>
  static std::unique_ptr<DataDistribution>
  block_generator(
    const typename BlockGenerator<S>::generator_t& generator,
    S&& init_state) {

    return
      GeneratorDataDistribution<S>::make(
        generator,
        std::forward<S>(init_state));
  }
};

}

#endif // DATA_DISTRIBUTION_H_
