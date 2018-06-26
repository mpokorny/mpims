find_package(MPI REQUIRED)

include_directories(
  ${INCLUDE_DIRECTORIES} "../src")
link_libraries(
  ${LINK_LIBRARIES} GTest::GTest)

add_executable(
  CyclicGeneratorTest
  CyclicGeneratorTest.cc)
target_compile_options(
  CyclicGeneratorTest PUBLIC
  ${GTEST_CFLAGS_OTHER})

add_executable(
  BlockSequenceGeneratorTest
  BlockSequenceGeneratorTest.cc)
target_compile_options(
  BlockSequenceGeneratorTest PUBLIC
  ${GTEST_CFLAGS_OTHER})

add_executable(
  UnpartitionedGeneratorTest
  UnpartitionedGeneratorTest.cc)
target_compile_options(
  UnpartitionedGeneratorTest PUBLIC
  ${GTEST_CFLAGS_OTHER})

add_executable(
  DataDistributionTest
  DataDistributionTest.cc)
target_include_directories(
  DataDistributionTest PRIVATE
  ${MPI_CXX_INCLUDE_PATH})
target_compile_options(
  DataDistributionTest PUBLIC
  ${GTEST_CFLAGS_OTHER})

gtest_discover_tests(CyclicGeneratorTest)
gtest_discover_tests(BlockSequenceGeneratorTest)
gtest_discover_tests(UnpartitionedGeneratorTest)
gtest_discover_tests(DataDistributionTest)
