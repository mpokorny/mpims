/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef MS_ARRAY_H_
#define MS_ARRAY_H_

#include <algorithm>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>

#include <IndexBlock.h>
#include <MSColumns.h>

namespace mpims {

struct MSArray {

  MSArray() {
  }

  MSArray(
    std::vector<IndexBlockSequence<MSColumns> >&& blocks_,
    std::shared_ptr<std::complex<float> >& buffer_)
    : blocks(std::move(blocks_))
    , buffer(buffer_) {
  }

  std::vector<IndexBlockSequence<MSColumns> > blocks;

  std::shared_ptr<std::complex<float> > buffer;

  bool
  operator==(const MSArray& other) const {
    return (
      blocks == other.blocks
      && ((!buffer && !other.buffer)
          || (buffer && other.buffer
              && std::memcmp(
                buffer.get(),
                other.buffer.get(),
                num_elements() * sizeof(*buffer)))));
  }

  bool
  operator!=(const MSArray& other) const {
    return !operator==(other);
  }

  size_t
  num_elements() const {
    size_t result = 0;
    std::for_each(
      std::begin(blocks),
      std::end(blocks),
      [&result](const IndexBlockSequence<MSColumns>& ibs) {
        result += ibs.num_elements();
      });
    return result;
  }

};

}

#endif // MS_ARRAY_H_
