/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>
#include <complex>
#include <cmath>
#include <ctime>
#include <forward_list>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <getopt.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <mpims.h>
#include <ColumnAxis.h>
#include <DataDistribution.h>
#include <MSColumns.h>
#include <Reader.h>

using namespace mpims;

std::size_t
suffix_multiplier(const std::string& suffix) {
  constexpr std::size_t KB = 1000;
  constexpr std::size_t KiB = 2 << 10;
  constexpr std::size_t MB = 1000 * KB;
  constexpr std::size_t MiB = KiB << 10;
  constexpr std::size_t GB = 1000 * MB;
  constexpr std::size_t GiB = MiB << 10;
  constexpr std::size_t vis = sizeof(std::complex<float>);

  std::unordered_map<std::string, std::size_t> multipliers = {
    {"k", KB}, {"K", KB}, {"kB", KB}, {"KB", KB},
    {"ki", KiB}, {"Ki", KiB}, {"kiB", KiB}, {"KiB", KiB},
    {"m", MB}, {"M", MB}, {"mB", MB}, {"MB", MB},
    {"mi", MiB}, {"Mi", MiB}, {"miB", MiB}, {"MiB", MiB},
    {"g", GB}, {"G", GB}, {"gB", GB}, {"GB", GB},
    {"gi", GiB}, {"Gi", GiB}, {"giB", GiB}, {"GiB", GiB},
    {"v", vis}, {"V", vis}
  };
  if (suffix.empty())
    return vis;
  else
    return multipliers[suffix];
}

std::size_t
parse_buffer_size(const std::string& bufsz) {
  std::size_t idx;
  std::size_t sz = std::stoull(bufsz, &idx);
  return sz * suffix_multiplier(bufsz.substr(idx, std::string::npos));
}

template <unsigned N>
auto
parse_colspec(const char *sep, const std::string& colspec) {
  auto fld = colspec.rfind(sep);
  return std::tuple_cat(
    parse_colspec<N-1>(sep, colspec.substr(0, fld)),
    std::make_tuple(std::stoull(colspec.substr(fld + 1, std::string::npos))));
}

template <>
auto
parse_colspec<0>(const char *, const std::string& colspec) {
  return std::make_tuple(mscol(colspec));
}

std::forward_list<std::string>
colspec_tokens(const char *sep, const std::string& specs) {
  std::forward_list<std::string> result;
  std::size_t pos = 0;
  std::size_t sc = specs.find(sep, pos);
  while (sc != std::string::npos) {
    result.emplace_front(specs.substr(pos, sc - pos));
    pos = sc + 1;
    sc = specs.find(sep, pos);
  }
  result.emplace_front(specs.substr(pos, sc - pos));
  result.reverse();
  return result;
}

std::vector<ColumnAxisBase<MSColumns> >
parse_shape(
  const char *token_sep,
  const char *spec_sep,
  const std::string& shape) {

  auto tokens = colspec_tokens(token_sep, shape);
  std::vector<ColumnAxisBase<MSColumns> > result;
  auto inserter = std::back_inserter(result);
  std::transform(
    std::begin(tokens),
    std::end(tokens),
    inserter,
    [&spec_sep](auto& tok) {
      MSColumns col;
      unsigned long long sz;
      std::tie(col, sz) = parse_colspec<1>(spec_sep, tok);
      return ColumnAxisBase<MSColumns>(static_cast<unsigned>(col), sz);
    });
  return result;
}

std::vector<MSColumns>
parse_traversal(const char *token_sep, const std::string& traversal) {

  auto tokens = colspec_tokens(token_sep, traversal);
  std::vector<MSColumns> result;
  auto inserter = std::back_inserter(result);
  std::transform(
    std::begin(tokens),
    std::end(tokens),
    inserter,
    [&](auto& tok) {
      return std::get<0>(parse_colspec<0>("", tok));
    });
  return result;
}

std::unordered_map<MSColumns, DataDistribution>
parse_distribution(
  const char *token_sep,
  const char *spec_sep,
  const std::string& distribution) {

  auto tokens = colspec_tokens(token_sep, distribution);
  std::unordered_map<MSColumns, DataDistribution> result;
  std::for_each(
    std::begin(tokens),
    std::end(tokens),
    [&spec_sep, &result](auto& tok) {
      MSColumns col;
      unsigned long long np, blk;
      std::tie(col, np, blk) = parse_colspec<2>(spec_sep, tok);
      result[col] = DataDistribution{ np, blk };
    });
  return result;
}

bool
parse_options(
  int argc,
  char *argv[],
  const char *token_sep,
  const char *spec_sep,
  std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  std::vector<MSColumns>& traversal_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t& buffer_size,
  std::string& ms_path,
  bool& ms_buffer_order,
  bool& debug_log) {

  int opt;
  struct option long_options[] = {
    {"msshape", required_argument, &opt, 's'},
    {"order", required_argument, &opt, 'o'},
    {"grid", required_argument, &opt, 'g'},
    {"buffer", required_argument, &opt, 'b'},
    {"transpose", no_argument, nullptr, 't'},
    {"no-transpose", no_argument, nullptr, -'t'},
    {"verbose", no_argument, nullptr, 'v'},
    {"help", optional_argument, nullptr, 'h'}
  };

  std::ostringstream usage;
  usage << "Usage: " << argv[0] << std::endl
        << "  (--msshape | -s) <ms-shape>" << std::endl
        << "  (--order |-o) <traversal-order>" << std::endl
        << "  (--buffer | -b) <buffer-size>" << std::endl
        << "  [(--grid |-g) <distribution>]" << std::endl
        << "  [((--transpose | -t) | --no-transpose)]" << std::endl
        << "  [(--verbose | -v)]" << std::endl
        << "  <ms-data-column-file>" << std::endl;

  ms_buffer_order = true;
  debug_log = false;
  bool got_shape = false, got_order = false, got_buffer = false;
  ms_path = "";

  while (1) {
    opt = 0;
    int c = getopt_long(argc, argv, "s:o:g::b:tvh", long_options, nullptr);

    if (c == -1) {
      ms_path = argv[optind];
      break;
    }

    int current_optind = optind - 1;

    switch (c) {
    case 0: {
      auto eq = strchr(argv[current_optind], '=');
      auto val = (eq ? eq + 1 : argv[current_optind]);

      switch (opt) {
      case 's':
        try {
          ms_shape = parse_shape(token_sep, spec_sep, val);
          got_shape = true;
        } catch (const std::exception& e) {
          std::cerr << "Failed to parse MS shape: "
                    << e.what() << std::endl;
        }
        break;

      case 'o':
        try {
          traversal_order = parse_traversal(token_sep, val);
          got_order = true;
        } catch (const std::exception& e) {
          std::cerr << "Failed to parse traversal order: "
                    << e.what() << std::endl;
        }
        break;

      case 'b':
        try {
          buffer_size = parse_buffer_size(val);
          got_buffer = true;
        } catch (const std::exception& e) {
          std::cerr << "Failed to parse buffer size: "
                    << e.what() << std::endl;
        }
        break;

      case 'g':
        try {
          pgrid = parse_distribution(token_sep, spec_sep, val);
        } catch (const std::exception& e) {
          std::cerr << "Failed to parse grid: "
                    << e.what() << std::endl;
          std::cout << usage.str();
          return false;
        }
        break;
      };
      break;
    }

    case 't':
      ms_buffer_order = false;
      break;

    case -'t':
      ms_buffer_order = true;
      break;

    case 'v':
      debug_log = true;
      break;

    case '?':
    case 'h':
      std::cout << usage.str();
      return false;
    }
  }

  if (!got_shape || !got_order || !got_buffer || ms_path == "") {
    std::cout << usage.str();
    return false;
  }

  return true;
}

unsigned
read_all(
  std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  std::vector<MSColumns>& traversal_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t buffer_size,
  std::string ms_path,
  bool ms_buffer_order,
  bool debug_log) {

  unsigned result = 0;

  try {
    auto reader =
      Reader::begin(
        ms_path,
        MPI_COMM_WORLD,
        MPI_INFO_NULL,
        ms_shape,
        traversal_order,
        ms_buffer_order,
        pgrid,
        buffer_size,
        debug_log);
    result = reader.num_ranks();
    while (reader != Reader::end()) {
      const MSArray& array __attribute__((unused)) = *reader;
      ++reader;
    }
  } catch (std::exception& e) {
    std::cerr << "Execution failed: " << e.what() << std::endl;
  }
  return result;
}

double
rnd(double t, double precision) {
  return std::round(t / precision) * precision;
}

double
ms(double t) {
  return rnd(t, 1.0e-3);
}

double
timespecd(const struct timespec& ts) {
  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

double
diff_timespec_ms(const struct timespec& t1, const struct timespec& t0) {
  return ms(timespecd(t1) - timespecd(t0));
}

double
timevald(const struct timeval& ts) {
  return ts.tv_sec + 1.0e-6 * ts.tv_usec;
}

double
diff_timeval_ms(const struct timeval& t1, const struct timeval& t0) {
  return ms(timevald(t1) - timevald(t0));
}

struct Times {
  double real, user, system;
};

template <typename F, typename ...Args>
std::tuple<std::invoke_result_t<F, Args...>, Times>
timeit(const F& f, Args...args) {
  typedef std::invoke_result_t<F, Args...> A;
  struct timespec ts[2];
  struct rusage ru[2];

  clock_gettime(CLOCK_MONOTONIC_COARSE, &ts[0]);
  getrusage(RUSAGE_SELF, &ru[0]);
  A a = f(args...);
  getrusage(RUSAGE_SELF, &ru[1]);
  clock_gettime(CLOCK_MONOTONIC_COARSE, &ts[1]);
  double real = diff_timespec_ms(ts[1], ts[0]);
  double user = diff_timeval_ms(ru[1].ru_utime, ru[0].ru_utime);
  double system = diff_timeval_ms(ru[1].ru_stime, ru[0].ru_stime);
  return std::make_tuple(a, Times{real, user, system});
}

int
main(int argc, char *argv[]) {

  const char token_separator[] = ",";
  const char spec_separator[] = ":";

  std::vector<ColumnAxisBase<MSColumns> > ms_shape;
  std::vector<MSColumns> traversal_order;
  std::unordered_map<MSColumns, DataDistribution> pgrid;
  std::size_t max_buffer_size;
  std::string ms_path;
  bool ms_buffer_order;
  bool debug_log;

  bool options_ok =
    parse_options(
      argc,
      argv,
      token_separator,
      spec_separator,
      ms_shape,
      traversal_order,
      pgrid,
      max_buffer_size,
      ms_path,
      ms_buffer_order,
      debug_log);

  if (options_ok) {
    ::MPI_Init(&argc, &argv);
    ::MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    mpi_call(::MPI_Barrier, MPI_COMM_WORLD);
    Times times;
    unsigned num_ranks;
    std::tie(num_ranks, times) =
      timeit(
        [&]() {
          unsigned n =
            read_all(
              ms_shape,
              traversal_order,
              pgrid,
              max_buffer_size,
              ms_path,
              ms_buffer_order,
              debug_log);
          mpi_call(::MPI_Barrier, MPI_COMM_WORLD);
          return n;
        });

    int rank;
    mpi_call(::MPI_Comm_rank, MPI_COMM_WORLD, &rank);
    std::array<double,2> tarray{ times.user, times.system };
    mpi_call(
      ::MPI_Reduce,
      (rank == 0) ? MPI_IN_PLACE : tarray.data(),
      tarray.data(),
      tarray.size(),
      MPI_DOUBLE,
      MPI_SUM,
      0,
      MPI_COMM_WORLD);
    if (rank == 0)
      std::cout << num_ranks << " reader processes" << std::endl
                << "real time: " << times.real << " sec" << std::endl
                << "total user time: " << tarray[0] << " sec" << std::endl
                << "total system time: " << tarray[1] << " sec" << std::endl;

    ::MPI_Finalize();
  }
}
