#ifndef BLOCK_GENERATOR_H_
#define BLOCK_GENERATOR_H_

#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace mpims {

typedef std::tuple<std::size_t, std::optional<std::size_t> > block_t;

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

  static constexpr auto
  initial_states(
    std::size_t block_length,
    std::size_t order,
    const std::optional<std::size_t>& axis_length) {

    return [=](std::size_t rank) {
      if (rank >= order)
        throw std::domain_error("rank is greater than or equal to order");

      std::size_t offset = rank * block_length;

      return
        State{
        block_length,
          block_length * order,
          std::min(offset, axis_length.value_or(offset)),
          axis_length};
    };
  }

  static constexpr std::size_t
  period(std::size_t block_length, std::size_t order) {
    return block_length * order;
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

  static auto
  is_unbounded(
    const std::vector<std::vector<block_t> >& all_blocks,
    const std::optional<std::size_t>& axis_length) {

    if (axis_length)
      return false;

    bool result = false;
    for (auto& blocks : all_blocks) {
      if (blocks.size() > 0) {
        auto brep =
          std::find_if(
            std::begin(blocks),
            std::end(blocks),
            [](auto& b) { return std::get<1>(b) == 0; });

        if (!axis_length) {
          if (brep == std::end(blocks)) {
            std::optional<std::size_t> blen;
            std::tie(std::ignore, blen) = blocks[blocks.size() - 1];
            result = result || !blen;
          } else {
            result = true;
          }
        }
      }
    }
    return result;
  }

  static auto
  initial_states(
    const std::vector<std::vector<block_t> >& all_blocks,
    const std::optional<std::size_t>& axis_length) {

    return [=](std::size_t rank) {
      if (rank >= all_blocks.size())
        throw std::domain_error("rank is greater than or equal to order");

      auto blocks = &all_blocks[rank];

      block_t prev = (*blocks)[0];
      for (std::size_t i = 1; i < blocks->size(); ++i) {
        std::size_t p0;
        std::optional<std::size_t> plen;
        std::tie(p0, plen) = prev;
        std::optional<std::size_t> prev_end =
          (plen ? (p0 + plen.value()) : plen);
        block_t cur = (*blocks)[i];
        if (!prev_end || std::get<0>(cur) < prev_end.value())
          throw std::domain_error("overlapping blocks");
        prev = cur;
      }

      auto brep =
        std::find_if(
          std::begin(*blocks),
          std::end(*blocks),
          [](auto& b) { return std::get<1>(b) == 0; });

      auto alen = axis_length;
      if (!alen && brep == std::end(*blocks) && blocks->size() > 0) {
        std::size_t b0;
        std::optional<std::size_t> blen;
        std::tie(b0, blen) = (*blocks)[blocks->size() - 1];
        alen = (blen ? (b0 + blen.value()) : blen);
      };

      if (brep != std::end(*blocks))
        ++brep;

      return
        State{std::vector<block_t>(std::begin(*blocks), brep), alen, 0, 0};
    };
  }

  static std::optional<std::size_t>
  period(const std::vector<std::vector<block_t> >& all_blocks) {

    std::vector<std::size_t > ps;
    std::for_each(
      std::begin(all_blocks),
      std::end(all_blocks),
      [&ps](auto& blks) {
        auto brep =
          std::find_if(
            std::begin(blks),
            std::end(blks),
            [](auto& b) { return std::get<1>(b) == 0; });
        if (brep != std::end(blks))
          ps.push_back(std::get<0>(*brep));
      });
    if (ps.size() == all_blocks.size()) {
      std::size_t result = 1;
      std::for_each(
        std::begin(ps),
        std::end(ps),
        [&result](auto& p) { result = std::lcm(result, p); });
      return result;
    } else {
      return std::nullopt;
    }
  }

  static std::tuple<State, std::optional<block_t> >
  apply(const State& st) {

    if (st.block_index >= st.blocks.size())
      return std::make_tuple(st, std::nullopt);

    std::size_t b0;
    std::optional<std::size_t> blen;
    std::tie(b0, blen) = st.blocks[st.block_index];
    b0 += st.block_offset;
    if (st.axis_length && b0 >= st.axis_length.value())
      return std::make_tuple(st, std::nullopt);

    if (st.axis_length) {
      if (blen)
        blen = std::min(b0 + blen.value(), st.axis_length.value()) - b0;
      else
        blen = st.axis_length.value() - b0;
    }
    auto next_blk = st.block_index + 1;
    auto next_blk_offset = st.block_offset;
    std::size_t next_b0;
    std::optional<std::size_t> next_blen;
    std::tie(next_b0, next_blen) = st.blocks[next_blk];
    if (next_blen == 0) {
      next_blk_offset += next_b0;
      if (!st.axis_length
          || (next_blk_offset + std::get<0>(st.blocks[0])
              < st.axis_length.value()))
        next_blk = 0;
    }
    return std::make_tuple(
      State{st.blocks, st.axis_length, next_blk, next_blk_offset},
      std::make_tuple(b0, blen));
  }
};

class UnpartitionedGenerator {

public:

  typedef block_t State;

  static auto constexpr
  initial_states(const std::optional<std::size_t>& axis_length) {
    return [=](const std::size_t rank) {
      if (rank > 0)
        throw std::domain_error("rank is greater than or equal to order");

      return block_t(0, axis_length);
    };
  }

  static std::tuple<State, std::optional<block_t> >
  apply(const State& st) {
    if (std::get<1>(st).value_or(1) > 0)
      return std::make_tuple(block_t(0, 0), st);
    else
      return std::make_tuple(st, std::nullopt);
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
