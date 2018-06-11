find_package(PkgConfig)
pkg_check_modules(GTEST REQUIRED gtest)

link_directories(
  ${LINK_DIRECTORIES} ${GTEST_LIBRARY_DIRS})
include_directories(
  ${INCLUDE_DIRECTORIES} ${GTEST_INCLUDE_DIRS} "../src")
# I don't know how to get the libpthread dependency out of GTEST_LDFLAGS or
# GTEST_LDFLAGS_OTHER, but I know that the dependency exists, so I'll just put
# it here manually
link_libraries(
  ${LINK_LIBRARIES} ${GTEST_LIBRARIES} "pthread")

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
