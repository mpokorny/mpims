#ifndef MPIMS_H_
#define MPIMS_H_

#include <climits>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <mpi.h>

#if SIZE_MAX == UCHAR_MAX
# define MPIMS_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
# define MPIMS_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
# define MPIMS_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
# define MPIMS_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
# define MPIMS_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
# error "Unable to match size_t size with MPI datatype"
#endif

namespace mpims {

enum AMode {
  WriteOnly,
  ReadWrite
};

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

template <typename A, typename F>
constexpr std::optional<std::invoke_result_t<F, A> >
map(const std::optional<A>& oa, F fn) {
  if (oa)
    return fn(oa.value());
  else
    return std::nullopt;
}


#endif // MPIMS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
