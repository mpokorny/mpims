#include <mutex>
#include <unordered_set>

#include <mpims.h>

using namespace mpims;

MPI_Errhandler comm_exception_errhandler_instance;
std::once_flag comm_flag;
MPI_Errhandler file_exception_errhandler_instance;
std::once_flag file_flag;

template <typename H>
void
handle_exception_errhandler(H*, int* rc, ...) {
  throw mpi_error(*rc);
}

void
create_comm_exception_errhandler() {
  MPI_Comm_create_errhandler(
    handle_exception_errhandler<MPI_Comm>,
    &comm_exception_errhandler_instance);
}

void
create_file_exception_errhandler() {
  MPI_File_create_errhandler(
    handle_exception_errhandler<MPI_File>,
    &file_exception_errhandler_instance);
}

MPI_Errhandler
mpims::comm_throw_exception() {
  std::call_once(comm_flag, create_comm_exception_errhandler);
  return comm_exception_errhandler_instance;
}

MPI_Errhandler
mpims::file_throw_exception() {
  std::call_once(file_flag, create_file_exception_errhandler);
  return file_exception_errhandler_instance;
}

void
mpims::set_throw_exception_errhandler(MPI_Comm comm) {
  int rc = MPI_Comm_set_errhandler(comm, comm_throw_exception());
  if (rc != MPI_SUCCESS)
    throw mpi_error(rc);
}

void
mpims::set_throw_exception_errhandler(MPI_File file) {
  int rc = MPI_File_set_errhandler(file, file_throw_exception());
  if (rc != MPI_SUCCESS)
    throw mpi_error(rc);
}

bool
mpims::datatype_is_predefined(MPI_Datatype dt) {
  const std::unordered_set<MPI_Datatype> predefined {
    MPI_CHAR, MPI_SHORT, MPI_INT, MPI_LONG, MPI_LONG_LONG_INT,
      MPI_SIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_SHORT, MPI_UNSIGNED,
      MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG_LONG, MPI_FLOAT, MPI_DOUBLE,
      MPI_LONG_DOUBLE, MPI_WCHAR, MPI_C_BOOL, MPI_INT8_T, MPI_INT16_T,
      MPI_INT32_T, MPI_INT64_T, MPI_UINT8_T, MPI_UINT16_T,
      MPI_UINT32_T, MPI_UINT64_T, MPI_C_COMPLEX, MPI_C_DOUBLE_COMPLEX,
      MPI_C_LONG_DOUBLE_COMPLEX, MPI_BYTE, MPI_PACKED, MPI_AINT, MPI_OFFSET,
      MPI_COUNT, MPI_CXX_BOOL, MPI_CXX_FLOAT_COMPLEX, MPI_CXX_DOUBLE_COMPLEX,
      MPI_CXX_LONG_DOUBLE_COMPLEX
      };
  return predefined.count(dt);
}

std::unique_ptr<MPI_Datatype, DatatypeDeleter>
mpims::datatype(MPI_Datatype dt) {
  return std::unique_ptr<MPI_Datatype, DatatypeDeleter>(new MPI_Datatype(dt));
}
