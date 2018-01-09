/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef MS_ARRAY_H_
#define MS_ARRAY_H_

#include <complex>
#include <memory>
#include <vector>

#include <IndexBlock.h>
#include <MSColumns.h>

namespace mpims {

struct MSArray {

  std::vector<IndexBlockSequence<MSColumns> > blocks;

  std::shared_ptr<std::complex<float> > buffer;

};

}

#endif // MS_ARRAY_H_
