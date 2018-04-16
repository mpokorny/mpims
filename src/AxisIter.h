/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef AXIS_ITER_H_
#define AXIS_ITER_H_

#include <memory>

#include <IterParams.h>

namespace mpims {

// AxisIter contains the state of the iteration along an axis. It follows the
// iteration pattern defined for an axis by an IterParams value.
class AxisIter {
public:

  AxisIter(std::shared_ptr<const IterParams> params_, bool outer_at_data_)
    : params(params_)
    , index(params_->origin)
    , block(0)
    , at_data((!params_->max_blocks || params_->max_blocks.value() > 0)
              && outer_at_data_)
    , outer_at_data(outer_at_data_)
    , at_end(params_->max_blocks && params_->max_blocks.value() == 0) {
  }

  bool
  operator==(const AxisIter& rhs) const {
    return *params == *rhs.params && index == rhs.index && block == rhs.block
      && at_data == rhs.at_data && outer_at_data == rhs.outer_at_data
      && at_end == rhs.at_end;
  }

  bool
  operator!=(const AxisIter& rhs) const {
    return !(operator==(rhs));
  }

  std::shared_ptr<const IterParams> params;
  std::size_t index;
  std::size_t block;
  bool at_data;
  bool outer_at_data;
  bool at_end;

  void
  increment(std::size_t n=1) {
    while (n > 0 && !at_end) {
      block = index / params->stride;
      auto block_origin = params->origin + block * params->stride;
      bool terminal_block =
        (params->max_blocks
         ? (block == params->max_blocks.value() - 1)
         : false);
      auto block_len =
        (terminal_block ? params->terminal_block_len : params->block_len);
      ++index;
      if (!terminal_block && index - block_origin >= block_len) {
        ++block;
        index = params->origin + block * params->stride;
        block_origin = index;
        terminal_block = (
          params->max_blocks
          ? (block == params->max_blocks.value() - 1)
          : false);
        block_len =
          (terminal_block ? params->terminal_block_len : params->block_len);
      }
      at_data = outer_at_data && (index - block_origin < block_len);
      at_end =
        terminal_block
        && (index - block_origin == params->max_terminal_block_len);
      --n;
    }
  }

  std::optional<std::size_t>
  num_remaining() const {
    if (at_end || !at_data)
      return 0;
    auto accessible_length = params->accessible_length();
    if (accessible_length) {
      auto block_origin = params->origin + block * params->stride;
      return (accessible_length.value()
              - (block * params->block_len + index - block_origin));
    } else {
      return std::nullopt;
    }
  }

  void
  complete() {
    at_data = false;
    at_end = true;
  }
};

}

#endif // AXIS_ITER_H_
