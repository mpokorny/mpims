/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef DATA_DISTRIBUTION_H_
#define DATA_DISTRIBUTION_H_

#include <cstddef>

namespace mpims {

struct DataDistribution {
  std::size_t num_processes;
  std::size_t block_size;
};

}

#endif // DATA_DISTRIBUTION_H_
