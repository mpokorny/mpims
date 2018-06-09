find_library(GTEST gtest)
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/include")

set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} -pthread")

add_executable(
  CyclicGeneratorTest
  CyclicGeneratorTest.cc
  "${GTEST_INCLUDE_DIR}/gtest/gtest.h")
target_include_directories(
  CyclicGeneratorTest PRIVATE
  ${GTEST_INCLUDE_DIR}
  "../src")
target_link_libraries(
  CyclicGeneratorTest
  gtest)
