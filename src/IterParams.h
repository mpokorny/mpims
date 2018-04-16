/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef ITER_PARAMS_H_
#define ITER_PARAMS_H_

#include <optional>

#include <MSColumns.h>

namespace mpims {

// IterParams holds the information for the iteration over a single axis, as
// well as context within the scope of iteration over the entire MS together
// given a user buffer size. Most of the values in this structure are used to
// describe the iteration over an axis for a particular process.
struct IterParams {
  MSColumns axis;
  // can a buffer hold the data for the full axis given iteration pattern?
  bool fully_in_array;
  // is iteration across this axis done without changing the fileview?
  bool within_fileview;
  // number of axis values in iteration for which the entire array comprising
  // data from all deeper axes can be copied into a single buffer
  std::size_t buffer_capacity;
  // number of data values in array comprising data from all deeper axes
  std::size_t array_length;
  // remainder are values describing iteration pattern
  std::size_t origin, stride, block_len,
    terminal_block_len, max_terminal_block_len;
  std::optional<std::size_t> length, max_blocks;

  bool
  operator==(const IterParams& rhs) const {
    return (
      axis == rhs.axis
      && fully_in_array == rhs.fully_in_array
      && within_fileview == rhs.within_fileview
      && buffer_capacity == rhs.buffer_capacity
      && length == rhs.length
      && origin == rhs.origin
      && stride == rhs.stride
      && block_len == rhs.block_len
      && max_blocks == rhs.max_blocks
      && terminal_block_len == rhs.terminal_block_len
      && max_terminal_block_len == rhs.max_terminal_block_len);
  }

  bool
  operator!=(const IterParams& rhs) const {
    return !(operator==(rhs));
  }

  std::optional<std::size_t>
  accessible_length() const {
    if (max_blocks.has_value())
      return block_len * (max_blocks.value() - 1) + terminal_block_len;
    else
      return std::nullopt;
  }

  std::optional<std::size_t>
  max_accessible_length() const {
    if (max_blocks.has_value())
      return block_len * (max_blocks.value() - 1) + max_terminal_block_len;
    else
      return std::nullopt;
  }
};

}

#endif // #define ITER_PARAMS_H_
