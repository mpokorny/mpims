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

typedef std::tuple<std::size_t, std::size_t> finite_block_t;

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
std::vector<finite_block_t>
blocks(const InputIterator& first, const InputIterator& last) {
  std::vector<finite_block_t> result;
  std::optional<finite_block_t> block;
  std::for_each(
    first,
    last,
    [&result, &block](const auto& i) {
      if (block) {
        auto blk = block.value();
        std::size_t b0 = 0;
        std::optional<std::size_t> blen = 0;
        std::tie(b0, blen) = blk;
        auto bend = b0 + blen.value();
        if (i > bend) {
          result.push_back(blk);
          block.emplace(i, 1);
        } else {
          assert(i == bend);
          block.emplace(b0, blen.value() + 1);
        }
      } else {
        block.emplace(i, 1);
      }
    });
  if (block)
    result.push_back(block.value());
  return result;
}

// DataDistribution
//
// abstract class for data access distributions
//
// use DataDistributionFactory to create instances
//
class DataDistribution
  : public std::enable_shared_from_this<DataDistribution> {

public:

  class Iterator
    : public std::iterator<std::input_iterator_tag, std::size_t> {

  public:

    virtual ~Iterator() {}

    virtual std::size_t
    operator*() const = 0;

    virtual Iterator&
    operator++() = 0;

    virtual bool
    at_end() const = 0;

    virtual bool
    is_unbounded() const = 0;

    std::vector<std::size_t>
    take(std::size_t n = 1) {
      std::vector<std::size_t> result;
      result.reserve(n);
      while (n-- > 0 && !at_end()) {
        result.push_back(operator*());
        operator++();
      }
      return result;
    }

    std::vector<finite_block_t>
    take_blocked(std::size_t n = 1) {
      auto seq = take(n);
      return mpims::blocks(std::begin(seq), std::end(seq));
    }

    std::vector<std::size_t>
    take_all() {
      if (is_unbounded())
        throw std::domain_error("not allowed on unbounded sequence");
      std::vector<std::size_t> result;
      while (!at_end()) {
        result.push_back(operator*());
        operator++();
      }
      return result;
    }

    std::vector<finite_block_t>
    take_all_blocked() {
      auto seq = take_all();
      return mpims::blocks(std::begin(seq), std::end(seq));
    }

    template <typename F>
    std::vector<size_t>
    take_while(F p) {
      std::vector<std::size_t> result;
      while (!at_end()) {
        auto i = operator*();
        if (p(i)) {
          result.push_back(i);
          operator++();
        } else {
          break;
        }
      }
      return result;
    }
  };

  virtual ~DataDistribution() {}

  virtual std::unique_ptr<Iterator>
  begin(std::size_t rank) const = 0;

  std::vector<finite_block_t>
  blocks(std::size_t rank) const {
    return begin(rank)->take_all_blocked();
  }

  std::optional<std::size_t>
  size(std::size_t rank) const {
    auto it = begin(rank);
    if (it->is_unbounded())
      return std::nullopt;
    else
      return it->take_all().size();
  }

  std::optional<std::size_t>
  max_size() const {
    std::optional<std::size_t> result = 0;
    for (std::size_t rank = 0; result && rank < m_order; ++rank) {
      auto sz = size(rank);
      if (sz)
        result = std::max(result.value(), sz.value());
      else
        result = sz;
    }
    return result;
  }

  std::optional<std::size_t>
  period() const {
    return m_period;
  }

  bool
  operator==(const DataDistribution& rhs) const {
    bool result = false;
    auto op = period();
    if (op) {
      if (op == rhs.period()) {
        auto ltp = [p = op.value()](std::size_t i) { return i < p; };
        for (std::size_t rank = 0; result && rank < m_order; ++rank)
          result =
            begin(rank)->take_while(ltp) == rhs.begin(rank)->take_while(ltp);
      }
    } else if (!rhs.period()) {
      result = true;
      for (std::size_t rank = 0; result && rank < m_order; ++rank)
        result = blocks(rank) == rhs.blocks(rank);
    }
    return result;
  }

  bool
  operator!=(const DataDistribution& rhs) const {
    return !operator==(rhs);
  }

  std::string
  show() const {
    std::ostringstream result;
    result << "{";
    const char *rksep = "";
    for (std::size_t rank = 0; rank < m_order; ++rank){
      result << rksep << rank << ":[";
      const char *blksep = "";
      for (const auto& blk : begin(rank)->take_all_blocked()) {
        std::size_t b0;
        std::optional<std::size_t> blen;
        std::tie(b0, blen) = blk;
        result << blksep
               << "(" << b0
               << "," << (blen ? std::to_string(blen.value()) : "-")
               << ")";
        blksep = ",";
      }
      result << "]";
      rksep = "; ";
    }
    result << "}";
    return result.str();
  }

  std::size_t
  order() const {
    return m_order;
  }

protected:

  std::size_t m_order;

  std::optional<std::size_t> m_period;
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
class GeneratorDataDistribution final
  : public DataDistribution {

public:

  typedef std::function<std::tuple<S, std::optional<block_t> >(const S&)>
  generator_t;

  typedef std::function<S(std::size_t)> initializer_t;

  // Note that the following static constructor methods return a new instance,
  // but in a shared_ptr rather than a unique_ptr. The reason is that an
  // Iterator instance pointer returned from DataDistribution::begin() holds a
  // pointer to the generator function in the DataDistribution instance, and
  // this pointer to the generator is implemented using an aliased shared_ptr.
  // This eliminates the need for users to explicitly maintain a reference to
  // the DataDistribution over the lifetime of an Iterator.

  static std::shared_ptr<DataDistribution>
  make(
    const generator_t& generator,
    const initializer_t& initializer,
    const std::optional<std::size_t>& period,
    std::size_t order,
    bool unbounded) {
    return std::make_shared<GeneratorDataDistribution>(
      generator,
      initializer,
      period,
      order,
      unbounded);
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
    m_initializer = std::move(other).m_initializer;
    m_period = std::move(other).m_period;
    m_order = std::move(other).m_order;
    m_unbounded = std::move(other).m_unbounded;
    return *this;
  }

  class GeneratorIterator final
    : public Iterator {

  public:

    GeneratorIterator(
      const std::shared_ptr<const generator_t>& generator,
      const S& state,
      bool unbounded)
      : m_generator(generator)
      , m_block_offset(0)
      , m_unbounded(unbounded) {

      std::tie(m_next_state, m_block) = (*m_generator)(state);
    }

    std::size_t
    operator*() const override {
      std::size_t b0;
      std::tie(b0, std::ignore) = m_block.value();
      return b0 + m_block_offset;
    }

    Iterator&
    operator++() override {
      std::size_t b0;
      std::optional<std::size_t> blen;
      std::tie(b0, blen) = m_block.value();
      if (blen && ++m_block_offset == blen.value()) {
        std::tie(m_next_state, m_block) = (*m_generator)(m_next_state);
        if (m_block && std::get<0>(m_block.value()) < b0 + blen.value())
          throw std::domain_error("overlapping blocks");
        m_block_offset = 0;
      }
      return *this;
    }

    bool
    at_end() const override {
      return !m_block;
    }

    bool
    is_unbounded() const override {
      return m_unbounded;
    }

  private:

    std::shared_ptr<const generator_t> m_generator;
    S m_next_state;
    std::optional<block_t> m_block;
    std::size_t m_block_offset;
    bool m_unbounded;
  };

  std::unique_ptr<Iterator>
  begin(std::size_t rank) const override {
    if (rank >= m_order)
      throw std::domain_error("rank is greater than or equal to order");
    std::shared_ptr<const generator_t> gen(shared_from_this(), &m_generator);
    return std::make_unique<GeneratorIterator>(
      gen,
      m_initializer(rank),
      m_unbounded);
  }

protected:

  void
  swap(GeneratorDataDistribution& other) {
    using std::swap;
    swap(m_generator, other.m_generator);
    swap(m_initializer, other.m_initializer);
    swap(m_period, other.m_period);
    swap(m_order, other.m_order);
    swap(m_unbounded, other.m_unbounded);
  }

public:

  // use of the static make() functions is preferred to use of the following
  // constructors
  GeneratorDataDistribution(
    const generator_t& generator,
    const initializer_t& initializer,
    const std::optional<std::size_t>& period,
    std::size_t order,
    bool unbounded)
    : m_generator(generator)
    , m_initializer(initializer)
    , m_unbounded(unbounded) {

    m_order = order;
    m_period = period;
  }

  GeneratorDataDistribution(const GeneratorDataDistribution& other)
    : m_generator(other.m_generator)
    , m_initializer(other.m_initializer)
    , m_unbounded(other.m_unbounded) {

    m_order = other.m_order;
    m_period = other.m_period;
  }

  GeneratorDataDistribution(GeneratorDataDistribution&& other)
    : m_generator(std::move(other).m_generator)
    , m_initializer(std::move(other).m_initializer)
    , m_unbounded(std::move(other).m_unbounded) {

    m_order = std::move(other).m_order;
    m_period = std::move(other).m_period;
  }

private:

  generator_t m_generator;

  initializer_t m_initializer;

  bool m_unbounded;
};

class DataDistributionFactory {

public:

  // block-cyclic data distribution
  //
  // * an empty 'axis_length' is used to indicate indefinite repetition
  //
  static std::shared_ptr<const DataDistribution>
  cyclic(
    std::size_t block_length,
    std::size_t order,
    const std::optional<std::size_t>& axis_length) {

    return
      GeneratorDataDistribution<CyclicGenerator::State>::make(
        CyclicGenerator::apply,
        CyclicGenerator::initial_states(block_length, order, axis_length),
        CyclicGenerator::period(block_length, order),
        order,
        !axis_length);
  }

  // (enumerated) block sequence data distribution
  //
  // * the number of sequence vectors is equal to the distribution order
  //
  // * a block of length 0 at some rank is used to indicate the start of a
  //   repetition for that rank (any blocks in the sequence following such a
  //   block are ignored)
  //
  // * an empty 'axis_length' is used to indicate an indefinite axis length
  //
  static std::shared_ptr<const DataDistribution>
  block_sequence(
    const std::vector<std::vector<block_t> >& all_blocks,
    const std::optional<std::size_t>& axis_length) {

    return
      GeneratorDataDistribution<BlockSequenceGenerator::State>::make(
        BlockSequenceGenerator::apply,
        BlockSequenceGenerator::initial_states(all_blocks, axis_length),
        BlockSequenceGenerator::period(all_blocks),
        all_blocks.size(),
        BlockSequenceGenerator::is_unbounded(all_blocks, axis_length));
  }

  static std::shared_ptr<const DataDistribution>
  unpartitioned(const std::optional<std::size_t>& axis_length) {

    return GeneratorDataDistribution<UnpartitionedGenerator::State>::make(
      UnpartitionedGenerator::apply,
      UnpartitionedGenerator::initial_states(axis_length),
      1,
      1,
      !axis_length);
  }

  // generic data distribution factory method
  //
  template <typename S>
  static std::shared_ptr<const DataDistribution>
  block_generator(
    const typename GeneratorDataDistribution<S>::generator_t& generator,
    const typename GeneratorDataDistribution<S>::initializer_t& initializer,
    std::size_t order) {

    return GeneratorDataDistribution<S>::make(generator, initializer, order);
  }
};

}

#endif // DATA_DISTRIBUTION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
