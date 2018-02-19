/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef MPIMS_H_
#define MPIMS_H_

#include <memory>
#include <stdexcept>
#include <type_traits>

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
    MPI_Error_string(errorcode, errorstr, &resultlen);
    // TODO: check result of last call
    return std::string(errorstr);
  }
};

MPI_Errhandler
comm_throw_exception();

MPI_Errhandler
file_throw_exception();

void
set_throw_exception_errhandler(MPI_Comm comm);

void
set_throw_exception_errhandler(MPI_File file);

bool
datatype_is_predefined(MPI_Datatype dt);

struct DatatypeDeleter {
  void operator()(MPI_Datatype* dt) {
    if (*dt != MPI_DATATYPE_NULL && !datatype_is_predefined(*dt)) {
      int rc = MPI_Type_free(dt);
      if (rc != MPI_SUCCESS)
        throw mpi_error(rc);
    }
    delete dt;
  }
};

std::unique_ptr<MPI_Datatype, DatatypeDeleter>
datatype(MPI_Datatype dt = MPI_DATATYPE_NULL);

} // end namespace mpims

template <
  typename T,
  typename = typename std::enable_if<std::is_unsigned<T>::value>::type >
T
ceil(T num, T denom) {
  return (num + (denom - 1)) / denom;
}


#endif // MPIMS_H_

