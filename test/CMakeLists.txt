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

gtest_discover_tests(CyclicGeneratorTest)
gtest_discover_tests(BlockSequenceGeneratorTest)
