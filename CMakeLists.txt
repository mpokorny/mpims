cmake_minimum_required(VERSION 2.8)

project(mpims)
set(MPIMS_VERSION_MAJOR 0)
set(MPIMS_VERSION_MINOR 1)
set(MPIMS_VERSION_PATCH 1)
set(MPIMS_VERSION
  ${MPIMS_VERSION_MAJOR}.${MPIMS_VERSION_MINOR}.${MPIMS_VERSION_PATCH})

set(
  CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

if (NOT DEFINED GTEST_ROOT)
  if (DEFINED ENV{GTEST_ROOT})
    set(GTEST_ROOT $ENV{GTEST_ROOT})
  else()
    message(FATAL_ERROR "GTEST_ROOT environment variable must be set")
  endif()
endif()


# set standard compiler flags
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17 -Og -g")
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -Ofast")

add_subdirectory(src)
