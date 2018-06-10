#include <algorithm>
#include <complex>
#include <cmath>
#include <ctime>
#include <forward_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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
#include <Writer.h>

using namespace mpims;


std::tuple<std::size_t, bool>
suffix_multiplier(const std::string& suffix) {
  constexpr std::size_t KB = 1000;
  constexpr std::size_t KiB = 2 << 10;
  constexpr std::size_t MB = 1000 * KB;
  constexpr std::size_t MiB = KiB << 10;
  constexpr std::size_t GB = 1000 * MB;
  constexpr std::size_t GiB = MiB << 10;
  constexpr std::size_t vis = 1;

  std::unordered_map<std::string, std::tuple<std::size_t, bool> > multipliers = {
    {"k", {KB, true}}, {"K", {KB, true}},
    {"kB", {KB, true}}, {"KB", {KB, true}},
    {"ki", {KiB, true}}, {"Ki", {KiB, true}},
    {"kiB", {KiB, true}}, {"KiB", {KiB, true}},
    {"m", {MB, true}}, {"M", {MB, true}},
    {"mB", {MB, true}}, {"MB", {MB, true}},
    {"mi", {MiB, true}}, {"Mi", {MiB, true}},
    {"miB", {MiB, true}}, {"MiB", {MiB, true}},
    {"g", {GB, true}}, {"G", {GB, true}},
    {"gB", {GB, true}}, {"GB", {GB, true}},
    {"gi", {GiB, true}}, {"Gi", {GiB, true}},
    {"giB", {GiB, true}}, {"GiB", {GiB, true}},
    {"v", {vis, false}}, {"V", {vis, false}}
  };
  if (suffix.empty())
    return multipliers["v"];
  else
    return multipliers[suffix];
}

std::tuple<std::size_t, bool>
parse_buffer_size(const std::string& bufsz) {
  std::size_t idx;
  std::size_t sz = std::stoull(bufsz, &idx);
  std::size_t mult;
  bool absolute;
  std::tie(mult, absolute) =
    suffix_multiplier(bufsz.substr(idx, std::string::npos));
  return std::make_tuple(sz * mult, absolute);
}

class ColspecParseError
  : public std::runtime_error {
public:

  explicit ColspecParseError()
    : std::runtime_error("colspec parsing error") {
  }
};

template <unsigned N>
auto
parse_colspec(const char *sep, const std::string& colspec) {
  auto fld = colspec.rfind(sep);
  if (fld == std::string::npos)
    throw ColspecParseError();
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
      try {
        std::tie(col, np, blk) = parse_colspec<2>(spec_sep, tok);
      } catch (const ColspecParseError&) {
        blk = 1;
        std::tie(col, np) = parse_colspec<1>(spec_sep, tok);
      }
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
  bool& complex_valued,
  std::vector<MSColumns>& traversal_order,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::tuple<std::size_t, bool>& buffer_size,
  std::string& ms_path,
  std::string& datarep,
  bool& debug_log) {

  int opt;
  struct option long_options[] = {
    {"msshape", required_argument, &opt, 's'},
    {"complex", no_argument, nullptr, 'c'},
    {"real", no_argument, nullptr, 'r'},
    {"order", required_argument, &opt, 'o'},
    {"grid", required_argument, &opt, 'g'},
    {"buffer", required_argument, &opt, 'b'},
    {"datarep", required_argument, &opt, 'd'},
    {"verbose", no_argument, nullptr, 'v'},
    {"help", optional_argument, nullptr, 'h'}
  };

  std::ostringstream usage;
  usage << "Usage: " << argv[0] << std::endl
        << "  (--msshape | -s) <ms-shape>" << std::endl
        << "  (--order |-o) <traversal-order>" << std::endl
        << "  (--buffer | -b) <buffer-size>" << std::endl
        << "  [(--grid |-g) <distribution>]" << std::endl
        << "  [(--complex | -c | --real | -r)]" << std::endl
        << "  [(--verbose | -v)]" << std::endl
        << "  [(--datarep | -d) <datarep>]" << std::endl
        << "  <ms-data-column-file>" << std::endl;

  if (argc == 1) {
    std::cout << usage.str();
    return false;
  }

  debug_log = false;
  complex_valued = true;
  bool got_shape = false, got_order = false, got_buffer = false;
  ms_path = "";
  datarep = "native";

  while (1) {
    opt = 0;
    int c = getopt_long(argc, argv, "s:o:g:b:d:crvh", long_options, nullptr);

    if (c == -1) {
      ms_path = argv[optind];
      break;
    }

    int current_optind = optind - 1;

    if (c == 0)
      c = opt;
    auto eq = strchr(argv[current_optind], '=');
    auto val = (eq ? eq + 1 : argv[current_optind]);

    switch (c) {
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

    case 'd':
      datarep = val;
      break;

    case 'c':
      complex_valued = true;
      break;

    case 'r':
      complex_valued = false;
      break;

    case 'v':
      debug_log = true;
      break;

    case 'h':
    case '?':
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

bool
validate_options(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  bool complex_valued,
  const std::vector<MSColumns>& traversal_order,
  const std::string& datarep) {

  static std::unordered_set<std::string> valid_datarep {
    "native",
      "internal",
      "external32" };

  bool result = true;

  bool complex_in_ms_shape =
    std::any_of(
      std::begin(ms_shape),
      std::end(ms_shape),
      [](const auto& ax) {
        return ax.id() == MSColumns::complex;
      });
  bool complex_in_traversal_order =
    std::any_of(
      std::begin(traversal_order),
      std::end(traversal_order),
      [](const auto& col) {
        return col == MSColumns::complex;
      });

  if (!complex_valued) {
    if (complex_in_ms_shape || complex_in_traversal_order) {
      std::cerr << "real-valued MS data cannot have '"
                << mscol_nickname(MSColumns::complex)
                << "' element in 'msshape' or 'order' option values"
                << std::endl;
      result = false;
    }
  } else {
    if (complex_in_ms_shape != complex_in_traversal_order) {
      std::cerr << "'"
                << mscol_nickname(MSColumns::complex)
                << "' element must not appear in only one of "
                << "'msshape' and 'order' option values"
                << std::endl;
      result = false;
    }
  }

  if (valid_datarep.count(datarep) == 0) {
    std::cerr << "unsupported 'datarep' value" << std::endl;
    result = false;
  }

  return result;
}

template <typename T>
void
write_buffer(
  std::optional<T*> buffer,
  std::size_t buffer_length,
  const T& val) {

  if (!buffer)
    return;
  auto buff = buffer.value();
  while (buffer_length-- > 0)
    *buff++ = val;
}

template <typename T>
unsigned
write_loop(
  Writer<T>&& writer,
  const std::optional<std::size_t>& num_outer,
  T& val,
  const T& addend) {

  while ((!num_outer && writer != Writer<T>::end())
         || (num_outer
             && (writer.outer_min_index().value_or(num_outer.value())
                 < num_outer.value()))) {
    auto buffer_length = writer.buffer_length();
    if (buffer_length > 0) {
      MSArray<T> array(buffer_length);
      write_buffer(array.buffer(), buffer_length, val);
      val += addend;
      *writer = std::move(array);
    }
    ++writer;
  }
  return writer.num_ranks();
}

unsigned
write_all(
  std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  bool handle_as_complex,
  std::vector<MSColumns>& traversal_order,
  std::optional<std::size_t> num_outer,
  std::unordered_map<MSColumns, DataDistribution>& pgrid,
  std::size_t buffer_size,
  std::string ms_path,
  std::string datarep,
  bool debug_log) {

  unsigned result = 0;

  try {
    if (handle_as_complex) {
      std::complex<float> val{1.0, 53.0}, addend{31.0, 17.0};
      result =
        write_loop(
          CxFltWriter::begin(
            ms_path,
            datarep,
            (num_outer ? AMode::WriteOnly : AMode::ReadWrite),
            MPI_COMM_WORLD,
            MPI_INFO_NULL,
            ms_shape,
            traversal_order,
            pgrid,
            buffer_size,
            debug_log),
          num_outer,
          val,
          addend);
    } else {
      float val(1.0), addend(31.0);
      result =
        write_loop(
          FltWriter::begin(
            ms_path,
            datarep,
            (num_outer ? AMode::WriteOnly : AMode::ReadWrite),
            MPI_COMM_WORLD,
            MPI_INFO_NULL,
            ms_shape,
            traversal_order,
            pgrid,
            buffer_size,
            debug_log),
          num_outer,
          val,
          addend);
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

bool
in_order_traversal(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order) {

  bool result = ms_shape.size() == traversal_order.size();
  for (unsigned i = 0; result && i < ms_shape.size(); ++i)
    result = ms_shape[i].id() == traversal_order[i];
  return result;
}

int
main(int argc, char *argv[]) {

  const char token_separator[] = ",";
  const char spec_separator[] = ":";

  std::vector<ColumnAxisBase<MSColumns> > ms_shape;
  bool complex_valued;
  std::vector<MSColumns> traversal_order;
  std::unordered_map<MSColumns, DataDistribution> pgrid;
  std::tuple<std::size_t, bool> max_buffer_size;
  std::string ms_path;
  std::string datarep;
  bool debug_log;

  bool options_ok =
    parse_options(
      argc,
      argv,
      token_separator,
      spec_separator,
      ms_shape,
      complex_valued,
      traversal_order,
      pgrid,
      max_buffer_size,
      ms_path,
      datarep,
      debug_log);

  if (options_ok)
    options_ok =
      validate_options(ms_shape, complex_valued, traversal_order, datarep);

  if (options_ok) {

    std::size_t buffer_size;
    bool buffer_size_is_absolute;
    std::tie(buffer_size, buffer_size_is_absolute) = max_buffer_size;
    bool handle_as_complex =
      std::none_of(
        std::begin(traversal_order),
        std::end(traversal_order),
        [](const auto& col) {
          return col == MSColumns::complex;
        })
      && complex_valued;
    if (!buffer_size_is_absolute)
      buffer_size *=
        (handle_as_complex ? sizeof(std::complex<float>) : sizeof(float));

    MPI_Init(&argc, &argv);
    set_throw_exception_errhandler(MPI_COMM_WORLD);

    bool in_order = in_order_traversal(ms_shape, traversal_order);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      std::cout << "** "
                << (in_order ? "write-only" : "read-before-write")
                << " mode **" << std::endl;

    std::optional<std::size_t> num_outer;
    if (in_order) {
      num_outer = ms_shape[0].length();
      ms_shape[0] =
        ColumnAxisBase<MSColumns>(static_cast<unsigned>(ms_shape[0].id()));
    }

    Times times;
    unsigned num_ranks;
    MPI_Barrier(MPI_COMM_WORLD);
    std::tie(num_ranks, times) =
      timeit(
        [&]() {
          unsigned n =
            write_all(
              ms_shape,
              handle_as_complex,
              traversal_order,
              num_outer,
              pgrid,
              buffer_size,
              ms_path,
              datarep,
              debug_log);
          MPI_Barrier(MPI_COMM_WORLD);
          return n;
        });

    std::array<double,2> tarray{ times.user, times.system };
    MPI_Reduce(
      (rank == 0) ? MPI_IN_PLACE : tarray.data(),
      tarray.data(),
      tarray.size(),
      MPI_DOUBLE,
      MPI_SUM,
      0,
      MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << num_ranks << " writer process"
                << ((num_ranks > 1) ? "es" : "") << std::endl;
      std::cout << "real time: " << times.real << " sec" << std::endl;
      std::cout << "total user time: " << tarray[0] << " sec";
      if (num_ranks > 1)
        std::cout << " (" << tarray[0] / num_ranks << " sec avg)";
      std::cout << std::endl;
      std::cout << "total system time: " << tarray[1] << " sec";
      if (num_ranks > 1)
        std::cout << " (" << tarray[1] / num_ranks << " sec avg)";
      std::cout << std::endl;
    }
    MPI_Finalize();
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
