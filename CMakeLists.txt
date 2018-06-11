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

# set standard compiler flags
set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -std=c++17 -Wall -Werror -Og")
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -Wall -Werror -Ofast")

add_subdirectory(src)
add_subdirectory(test)
