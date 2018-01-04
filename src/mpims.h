/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef MPIMS_H_
#define MPIMS_H_

#include <stdexcept>

#include <mpi.h>

namespace mpims {

class mpi_error
  : public std::runtime_error {

public:

  explicit mpi_error(int errorcode)
    : std::runtime_error(error_string(errorcode)) {
  }

private:
  static std::string
  error_string(int errorcode) {
    char errorstr[MPI_MAX_ERROR_STRING];
    int resultlen;
    ::MPI_Error_string(errorcode, errorstr, &resultlen);
    // TODO: check result of last call
    return std::string(errorstr);
  }
};

template <typename Function, typename ...Args>
void
mpi_call(Function fn, Args...args) {
  int rc = fn(args...);
  if (rc != MPI_SUCCESS)
    throw mpi_error(rc);
}

} // end namespace mpims

#endif // MPIMS_H_

