/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef PROCESS_DISTRIBUTION_H_
#define PROCESS_DISTRIBUTION_H_

namespace mpims {

struct ProcessDistribution {
  std::size_t num_processes;
  std::size_t block_size;
};

}

#endif // PROCESS_DISTRIBUTION_H_
