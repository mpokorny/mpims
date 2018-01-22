#include <unordered_set>

#include <mpims.h>

using namespace mpims;

bool
mpims::datatype_is_predefined(::MPI_Datatype dt) {
  const std::unordered_set<::MPI_Datatype> predefined {
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

std::unique_ptr<::MPI_Datatype, DatatypeDeleter>
mpims::datatype(::MPI_Datatype dt) {
  return std::unique_ptr<::MPI_Datatype, DatatypeDeleter>(
    new ::MPI_Datatype(dt));
}
