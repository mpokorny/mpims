# mpims #

Exploration of parallel I/O access to
[casacore](https://casacore.github.io/casacore/) Measurement Set data using
[MPI-IO](http://mpi-forum.org/docs/mpi-3.1/mpi31-report/node305.htm#Node305).

## Scope  ##

The current scope for this code is somewhat limited, primarily because it exists
only to explore the potential costs and benefits of using parallel I/O to access
Measurement Set data. The code is not intended to provide full-featured access
to an MS, and provides no interface to casacore itself, as at this point the
methods being investigated here cannot be integrated into casacore directly (for
example, as a casacore Table System DataManager). Certain of the limitations of
this code might be relatively easily removed, but as the code in this project is
not required to meet any external requirements, its development is completely
subject to the interests of the authors.

The major limitation for this project from a casacore perspective is that it is
developed only to access MS main table data columns. Access to MS sub-tables is
out of scope, which leads to lack of support for data selections, for example.
Access to flag columns could be supported in principle, but that is not
currently implemented, as the initial objective is to profile I/O performance,
and supporting flag data is not necessary for that purpose.

## Goals ##

Cheating here a bit, as the following list is mostly based on the current code,
but this provides a place for future additions.

- Support reading and writing of MS main table data columns using MPI-IO.
- Support maximum flexibility in access patterns, for both reading and writing; in
  particular, read patterns are independent of write patterns.
- Support parallelization in a bulk synchronous parallel program framework,
  based on MPI.
- Provide flexible parallelization patterns, along arbitrary axes, using
  user-defined, chunked, round-robin data distribution.
- Provide an array-oriented style of data access, subject to user-defined buffer
  sizes and access patterns.
- Provide for asynchronous, non-blocking data column access. Provide
  thread-safety for threaded user code.

### Status ###

All of the goals above are satisfied by the current code. The writing
functionality is currently only available in synchronous mode.

## Possible extensions ##

- Support access to tiled MS data columns, such as those created by a
  TiledColumnStMan instance.
- Support access to non-uniform MS data columns, such as those created by a
  TiledDataStMan instance.

## Build instructions ##

Build is based on cmake. No special flags are required, although do be careful
to check the MPI compiler and library found by cmake, and provide needed flags
in case an undesired version is found. A cmake command script is included in the
project for reference, but it is almost certainly not usable as-is, having been
created for the main developer's own environment.

### Requirements ###
- cmake
- MPI library, built with support for the targeted file systems
- g++-7 (mpims uses c++17 features)

## Build artifacts ##

### libmpims

Library for reading and writing (uniform) MS data columns using MPI-IO.

### reader-test

A test suite for the reader class of *libmpims*, `mpims::Reader`. This test
suite is not automated, and it includes test configurations for two data
distributions (among processes), which can only be selected by modifying source
code and recompiling. Additionally, although various failures are clearly
indicated in the output, there are some failure modes that are not so obvious,
and require a careful reading of the output to find. Nevertheless, I have found
this test suite to be very useful.

### writer-test

A test suite for the writer class of *libmpims*, `mpims::Writer`. Similar
caveats apply to this test suite as for *reader-test*.

### preadcol

An application for reading any uniform MS data column in parallel, with
reporting of the time required to complete. Many options are supported...*which
will be documented here eventually*.

### pwritecol

An application for writing any uniform MS data column in parallel, with
reporting of the time required to complete. Many options are supported...*which
will be documented here eventually*.
