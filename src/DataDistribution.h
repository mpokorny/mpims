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

    friend class IteratorDataDistribution<T>;

    friend std::unique_ptr<IteratorDataDistributionIterator>
    std::make_unique<IteratorDataDistributionIterator>(const T&);

  public:

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

  protected:

    IteratorDataDistributionIterator(const T& iter)
      : m_iter(iter) {
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

  IteratorDataDistribution(iterator&& begin)
    : m_begin(std::forward<iterator>(begin)) {
  }

  IteratorDataDistribution(const IteratorDataDistribution& other)
    : m_begin(other.m_begin) {
  }

  IteratorDataDistribution(IteratorDataDistribution&& other)
    : m_begin(std::move(other).m_begin) {
  }

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
// data distribution for a generator
//
// this is the implementation sub-class of DataDistribution, which should
// support almost all sorts of data distributions (all that's required is a
// generator function and an initial state)
//
template <typename S>
class BlockGeneratorDataDistribution
  : public IteratorDataDistribution<BlockGeneratorIterator<S> > {

  typedef typename BlockGeneratorIterator<S>::generator_t generator_t;

  friend std::unique_ptr<BlockGeneratorDataDistribution>
  std::make_unique<BlockGeneratorDataDistribution>(
    const generator_t& generator,
    S&&);

private:

  BlockGeneratorDataDistribution(const generator_t& generator, S&& state)
    : IteratorDataDistribution<BlockGeneratorIterator<S> >(
      BlockGeneratorIterator(generator, std::forward<S>(state))) {
  }

public:

  static std::unique_ptr<DataDistribution>
  make(const generator_t& generator, S&& state) {
    return std::make_unique<BlockGeneratorDataDistribution>(
      generator,
      std::forward<S>(state));
  }

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
      BlockGeneratorDataDistribution<CyclicGenerator::State>::make(
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
      BlockGeneratorDataDistribution<BlockSequenceGenerator::State>::make(
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
      BlockGeneratorDataDistribution<
        typename BlockGenerator<InputIterator>::State>::make(
          BlockGenerator<InputIterator>::apply,
          typename BlockGenerator<InputIterator>::State{begin, end});
  }

  // generic data distribution factory method
  //
  template <typename S>
  static std::unique_ptr<DataDistribution>
  block_generator(
    const typename BlockGeneratorIterator<S>::generator_t& generator,
    S&& init_state) {

    return
      BlockGeneratorDataDistribution<S>::make(
        generator,
        std::forward<S>(init_state));
  }
};

}

#endif // DATA_DISTRIBUTION_H_
