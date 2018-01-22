/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef MPIMS_H_
#define MPIMS_H_

#include <memory>
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

bool
datatype_is_predefined(::MPI_Datatype dt);

struct DatatypeDeleter {
  void operator()(::MPI_Datatype* dt) {
    if (*dt != MPI_DATATYPE_NULL && !datatype_is_predefined(*dt))
      mpi_call(::MPI_Type_free, dt);
    delete dt;
  }
};

std::unique_ptr<::MPI_Datatype, DatatypeDeleter>
datatype(::MPI_Datatype dt = MPI_DATATYPE_NULL);

} // end namespace mpims

#endif // MPIMS_H_

