#!/bin/bash
BUILD_TYPE=Debug
if [ "$1" != "" ]; then
    BUILD_TYPE=$1
fi
GCC=/opt/local/compilers/gcc-7/bin/gcc
GXX=/opt/local/compilers/gcc-7/bin/g++
EXTRA_DEBUG_FLAGS=-gdwarf-3
MPICC=/opt/cbe-local/stow/mvapich2-2.3b-mp/bin/mpicc
MPICXX=/opt/cbe-local/stow/mvapich2-2.3b-mp/bin/mpicxx

BUILD_DIR=./build

CMAKE="mkdir -p $BUILD_DIR && cd $BUILD_DIR && \
~/stow/cmake-3.6.3/bin/cmake \
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_CXX_FLAGS_DEBUG=$EXTRA_DEBUG_FLAGS \
-DCMAKE_C_COMPILER=$GCC \
-DCMAKE_CXX_COMPILER=$GXX \
-DMPI_C_COMPILER=$MPICC \
-DMPI_CXX_COMPILER=$MPICXX .."

echo $CMAKE
eval $CMAKE
