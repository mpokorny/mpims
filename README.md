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
this test suite to be rather useful.

### writer-test

A test suite for the writer class of *libmpims*, `mpims::Writer`. Similar
caveats apply to this test suite as for *reader-test*.

### preadcol

An application for reading any uniform MS data column in parallel, with
reporting of the time required to complete. Required input is an un-tiled (or
trivially tiled) MS data column file. The command syntax may be reviewed by
executing *preadcol* without arguments, or with one of the flags `-h`, `--help`,
or `-?`.

``` shell
$ ./preadcol --help
Usage: ./preadcol
  (--msshape | -s) <ms-shape>
  (--order |-o) <traversal-order>
  (--buffer | -b) <buffer-size>
  [(--grid |-g) <distribution>]
  [((--transpose | -t) | --no-transpose)]
  [(--readahead | -r)]
  [(--verbose | -v)]
  [(--datarep | -d) <datarep>]
  <ms-data-column-file>
```
Required options are `--msshape`, `--order` and `--buffer`.
* `--msshape`: shape of MS data column; syntax is AX:N[,AX:N]*, where AX is
  the abbreviation of a column name, and N is number of elements on that axis;
  example: `TIM:360,SPW:16,BAL:378,CH:256,POL:4`
* `--order`: traversal order of MS data column; syntax is AX[,AX]*, where AX
  is the abbreviation of a column name; example: `TIM,SPW,BAL,CH,POL`
* `--buffer`: size of the buffer into which data values are read; syntax is size
  in bytes using standard abbreviations (*e.g.*, MB, GiB), or as a number of
  data elements
* `--grid`: distribution of data across process grid; syntax is AX:N[:B], where
  AX is that abbreviation of a column name, N is the number of processes along
  that axis, and B (default value 1) is the number of axis values in a block;
  distribution is round-robin by block on every distributed axis; example:
  `SPW:16`
* `--transpose`: boolean flag to specify whether values in the read buffer are
  transposed (relative to MS order) to match the traversal order or not; note
  that the value of this flag does *not* affect the traversal order, only the
  ordering of values within a single read buffer; default is `--no-transpose`,
  which generally provides better performance
* `--readahead`: boolean flag to specify whether the reading process should
  asynchronously read ahead of client requests by one buffer; not very useful
  for *preadcol* itself, but it is a feature of the underlying library
* `--verbose`: produce verbose debugging output
* `--datarep`: data value representation in file; value may by any of those
  supported by the MPI implementation (normally "native", "internal" and
  "external32")

Being an MPI program, *preadcol* should generally be started using an MPI job
launcher, although for a single process, that is not necessary.

### pwritecol

An application for writing any uniform MS data column in parallel, with
reporting of the time required to complete. The output file is an untiled MS
data column file. The command line options are fewer than for *preadcol*,
although for those that are available, their meaning and usage are identical to
those of *preadcol*. The command syntax may be reviewed by executing *pwritecol*
without arguments, or with one of the flags `-h`, `--help`, or `-?`.

``` shell
$ ./pwritecol --help
Usage: ./pwritecol
  (--msshape | -s) <ms-shape>
  (--order |-o) <traversal-order>
  (--buffer | -b) <buffer-size>
  [(--grid |-g) <distribution>]
  [(--verbose | -v)]
  [(--datarep | -d) <datarep>]
  <ms-data-column-file>
```

