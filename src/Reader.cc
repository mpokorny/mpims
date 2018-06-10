#include <complex>

#include <mpi.h>

#include <Reader.h>

using namespace mpims;

template <> MPI_Datatype CxFltReader::value_datatype = MPI_CXX_FLOAT_COMPLEX;
template <> std::size_t CxFltReader::value_size = sizeof(std::complex<float>);
template <> bool CxFltReader::is_complex = true;

template <> MPI_Datatype FltReader::value_datatype = MPI_FLOAT;
template <> std::size_t FltReader::value_size = sizeof(float);
template <> bool FltReader::is_complex = false;

template <> MPI_Datatype CxDblReader::value_datatype = MPI_CXX_DOUBLE_COMPLEX;
template <> std::size_t CxDblReader::value_size = sizeof(std::complex<double>);
template <> bool CxDblReader::is_complex = true;

template <> MPI_Datatype DblReader::value_datatype = MPI_DOUBLE;
template <> std::size_t DblReader::value_size = sizeof(double);
template <> bool DblReader::is_complex = false;

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
