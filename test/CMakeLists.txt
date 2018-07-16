find_package(MPI REQUIRED)

include_directories(
  ${INCLUDE_DIRECTORIES} "../src")

add_executable(
  CyclicGeneratorTest
  CyclicGeneratorTest.cc)
target_compile_options(
  CyclicGeneratorTest PUBLIC
  ${GTEST_CFLAGS_OTHER})
target_link_libraries(
  CyclicGeneratorTest
  GTest::Main)

add_executable(
  BlockSequenceGeneratorTest
  BlockSequenceGeneratorTest.cc)
target_compile_options(
  BlockSequenceGeneratorTest PUBLIC
  ${GTEST_CFLAGS_OTHER})
target_link_libraries(
  BlockSequenceGeneratorTest
  GTest::Main)

add_executable(
  UnpartitionedGeneratorTest
  UnpartitionedGeneratorTest.cc)
target_compile_options(
  UnpartitionedGeneratorTest PUBLIC
  ${GTEST_CFLAGS_OTHER})
target_link_libraries(
  UnpartitionedGeneratorTest
  GTest::Main)

add_executable(
  DataDistributionTest
  DataDistributionTest.cc)
target_include_directories(
  DataDistributionTest PRIVATE
  ${MPI_CXX_INCLUDE_PATH})
target_compile_options(
  DataDistributionTest PUBLIC
  ${GTEST_CFLAGS_OTHER})
target_link_libraries(
  DataDistributionTest
  GTest::Main)

gtest_discover_tests(CyclicGeneratorTest)
gtest_discover_tests(BlockSequenceGeneratorTest)
gtest_discover_tests(UnpartitionedGeneratorTest)
gtest_discover_tests(DataDistributionTest)
