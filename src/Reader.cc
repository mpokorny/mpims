/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <mpims.h>

#include <ArrayIndexer.h>
#include <Reader.h>

using namespace mpims;

template <> MPI_Datatype Reader<std::complex<float> >::value_datatype =
  MPI_CXX_FLOAT_COMPLEX;

template <> std::size_t Reader<std::complex<float> >::value_size =
  sizeof(std::complex<float>);

template <> MPI_Datatype Reader<float>::value_datatype = MPI_FLOAT;

template <> std::size_t Reader<float>::value_size = sizeof(float);

template <> MPI_Datatype Reader<std::complex<double> >::value_datatype =
  MPI_CXX_DOUBLE_COMPLEX;

template <> std::size_t Reader<std::complex<double> >::value_size =
  sizeof(std::complex<double>);

template <> MPI_Datatype Reader<double>::value_datatype = MPI_DOUBLE;

template <> std::size_t Reader<double>::value_size = sizeof(double);
