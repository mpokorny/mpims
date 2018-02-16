find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++17")

# use, i.e. don't skip, the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already (but later on when
# installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

# mpims library
add_library(mpims SHARED
  mpims.cc
  MSColumns.cc
  MSArray.cc
  Reader.cc
  Writer.cc)
target_include_directories(mpims PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  .)
target_compile_options(mpims PRIVATE
  ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(mpims PRIVATE
  ${MPI_CXX_LIBRARIES})
set_target_properties(mpims PROPERTIES
  VERSION ${MPIMS_VERSION}
  SOVERSION ${MPIMS_VERSION_MAJOR})

# install rules

install(TARGETS mpims
  LIBRARY DESTINATION lib
  CONFIGURATIONS Release
  PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(FILES
  ArrayIndexer.h
  ColumnAxis.h
  DataDistribution.h
  IndexBlock.h
  mpims.h
  MSArray.h
  MSColumns.h
  Reader.h
  ReaderMPIState.h
  Writer.h
  DESTINATION include
  CONFIGURATIONS Release
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)

add_executable(reader-test
  reader-test.cc)
target_include_directories(reader-test PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  .)
target_compile_options(reader-test PRIVATE
  ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(reader-test
  ${MPI_CXX_LIBRARIES}
  mpims)

add_executable(preadcol
  preadcol.cc)
target_include_directories(preadcol PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  .)
target_compile_options(preadcol PRIVATE
  ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(preadcol
  ${MPI_CXX_LIBRARIES}
  mpims)

add_executable(writer-test
  writer-test.cc)
target_include_directories(writer-test PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  .)
target_compile_options(writer-test PRIVATE
  ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(writer-test
  ${MPI_CXX_LIBRARIES}
  mpims)

add_executable(pwritecol
  pwritecol.cc)
target_include_directories(pwritecol PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  .)
target_compile_options(pwritecol PRIVATE
  ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(pwritecol
  ${MPI_CXX_LIBRARIES}
  mpims)
