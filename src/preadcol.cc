/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>
#include <ctime>
#include <forward_list>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <getopt.h>
#include <unistd.h>
#include <sys/times.h>

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

  std::unordered_map<std::string, std::size_t> multipliers = {
    {"k", KB}, {"K", KB}, {"kB", KB}, {"KB", KB},
    {"ki", KiB}, {"Ki", KiB}, {"kiB", KiB}, {"KiB", KiB},
    {"m", MB}, {"M", MB}, {"mB", MB}, {"MB", MB},
    {"mi", MiB}, {"Mi", MiB}, {"miB", MiB}, {"MiB", MiB},
    {"g", GB}, {"G", GB}, {"gB", GB}, {"GB", GB},
    {"gi", GiB}, {"Gi", GiB}, {"giB", GiB}, {"GiB", GiB}
  };
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
  std::string& ms_path) {

  int opt;
  struct option long_options[] = {
    {"msshape", required_argument, &opt, 's'},
    {"order", required_argument, &opt, 'o'},
    {"grid", optional_argument, nullptr, 'g'},
    {"buffer", required_argument, &opt, 'b'},
    {"help", optional_argument, nullptr, 'h'}
  };

  std::ostringstream usage;
  usage << "Usage: " << argv[0] << std::endl
        << "  (--msshape | -s) <ms-shape>" << std::endl
        << "  (--order |-o) <traversal-order>" << std::endl
        << "  [(--grid |-g) <distribution>]" << std::endl
        << "  (--buffer | -b) <buffer-size>" << std::endl
        << "  <ms-data-column-file>" << std::endl;
  
  bool got_shape = false, got_order = false, got_buffer = false;
  ms_path = "";

  while (1) {
    opt = 0;
    int c = getopt_long(argc, argv, "s:o:g::b:h", long_options, nullptr);

    if (c == -1) {
      ms_path = argv[optind];
      break;
    }

    int current_optind = optind - 1;

    switch (c) {
    case 0:
      switch (opt) {
      case 's':
        try {
          auto eq = strchr(argv[current_optind], '=');
          ms_shape =
            parse_shape(token_sep, spec_sep, eq ? eq + 1 : argv[current_optind]);
          got_shape = true;
        } catch (const std::exception& e) {
          std::cerr << "Failed to parse MS shape: "
                    << e.what() << std::endl;
        }
        break;

      case 'o':
        try {
          auto eq = strchr(argv[current_optind], '=');
          traversal_order =
            parse_traversal(token_sep, eq ? eq + 1 : argv[current_optind]);
          got_order = true;
        } catch (const std::exception& e) {
          std::cerr << "Failed to parse traversal order: "
                    << e.what() << std::endl;
        }
        break;

      case 'b':
        try {
          auto eq = strchr(argv[current_optind], '=');
          buffer_size = parse_buffer_size(eq ? eq + 1: argv[current_optind]);
          got_buffer = true;
        } catch (const std::exception& e) {
          std::cerr << "Failed to parse buffer size: "
                    << e.what() << std::endl;
        }
        break;
      };
      break;

    case 'g':
      pgrid = parse_distribution(token_sep, spec_sep, argv[current_optind]);
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

void
read_all(
  std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  std::vector<MSColumns>& traversal_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t buffer_size,
  std::string ms_path) {

  auto reader =
    Reader::begin(
      ms_path,
      MPI_COMM_WORLD,
      MPI_INFO_NULL,
      ms_shape,
      traversal_order,
      pgrid,
      buffer_size);
  while (reader != Reader::end()) {
    const MSArray& array __attribute__((unused)) = *reader;
    ++reader;
  }
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
      ms_path);

  if (options_ok) {
    ::MPI_Init(&argc, &argv);
    ::MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank;
    mpi_call(::MPI_Comm_rank, MPI_COMM_WORLD, &rank);

    long clk_tck = sysconf(_SC_CLK_TCK);
    clock_t t0, t1;
    struct tms tms0, tms1;

    mpi_call(::MPI_Barrier, MPI_COMM_WORLD);
    t0 = times(&tms0);
    read_all(ms_shape, traversal_order, pgrid, max_buffer_size, ms_path);
    mpi_call(::MPI_Barrier, MPI_COMM_WORLD);
    t1 = times(&tms1);

    if (rank == 0){
      std::cout << "real time: "
        << static_cast<double>(t1 - t0) / clk_tck
                << " sec" << std::endl;
      std::cout << "user time: "
        << static_cast<double>(tms1.tms_utime - tms0.tms_utime) / clk_tck
                << " sec" << std::endl;
      std::cout << "system time: "
        << static_cast<double>(tms1.tms_stime - tms0.tms_stime) / clk_tck
                << " sec" << std::endl;
    }
    ::MPI_Finalize();
  }
}
