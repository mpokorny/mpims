/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <complex>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include <mpi.h>

#include <mpims.h>
#include <ColumnAxis.h>
#include <Reader.h>

using namespace mpims;

using namespace std;

constexpr auto ntim = 4;
constexpr auto nbits_tim = 2;
constexpr auto nspw = 4;
constexpr auto nbits_spw = 2;
constexpr auto nbal = 3 /*6*/;
constexpr auto nbits_bal = 3;
constexpr auto nch = 13;
constexpr auto nbits_ch = 4;
constexpr auto npol = 2;
constexpr auto nbits_pol = 1;

void
encode_vis(
  complex<float>& vis,
  size_t tim,
  size_t spw,
  size_t bal,
  size_t ch,
  size_t pol) {

  vis.real(static_cast<float>((((bal << nbits_spw) | spw) << nbits_tim) | tim));
  vis.imag(static_cast<float>((ch << nbits_pol) | pol));
}

void
decode_vis(
  const complex<float>& vis,
  size_t& tim,
  size_t& spw,
  size_t& bal,
  size_t& ch,
  size_t& pol) {

  auto re = static_cast<unsigned>(vis.real());
  tim = re & ((1 << nbits_tim) - 1);
  re >>= nbits_tim;
  spw = re & ((1 << nbits_spw) - 1);
  re >>= nbits_spw;
  bal = re;

  auto im = static_cast<unsigned>(vis.imag());
  pol = im & ((1 << nbits_pol) - 1);
  im >>= nbits_pol;
  ch = im;
}

bool
checkit(
  const complex<float>* buffer,
  size_t& n,
  vector<pair<MSColumns, size_t> >& coords,
  vector<IndexBlockSequence<MSColumns> >::const_iterator begin,
  vector<IndexBlockSequence<MSColumns> >::const_iterator end,
  ostringstream& output)
  __attribute__((unused));

bool
checkit(
  const complex<float>* buffer,
  size_t& n,
  vector<pair<MSColumns, size_t> >& coords,
  vector<IndexBlockSequence<MSColumns> >::const_iterator begin,
  vector<IndexBlockSequence<MSColumns> >::const_iterator end,
  ostringstream& output) {

  auto next = begin;
  ++next;
  bool result = true;
  if (next != end) {
    for (auto& b : begin->m_blocks)
      for (size_t i = 0; i < b.m_length; ++i) {
        coords.emplace_back(begin->m_axis, b.m_index + i);
        result = checkit(buffer, n, coords, next, end, output) && result;
        coords.pop_back();
      }
  } else {
    for (auto& b : begin->m_blocks)
      for (size_t i = 0; i < b.m_length; ++i) {
        coords.emplace_back(begin->m_axis, b.m_index + i);
        unordered_map<MSColumns, size_t> coords_map(
          std::begin(coords), std::end(coords));
        size_t tim, spw, bal, ch, pol;
        decode_vis(buffer[n++], tim, spw, bal, ch, pol);
        if (coords_map[MSColumns::time] != tim
            || coords_map[MSColumns::spectral_window] != spw
            || coords_map[MSColumns::baseline] != bal
            || coords_map[MSColumns::channel] != ch
            || coords_map[MSColumns::polarization_product] != pol) {
          result = false;
          unordered_map<MSColumns, size_t> value_map {
            {MSColumns::time, tim},
            {MSColumns::spectral_window, spw},
            {MSColumns::baseline, bal},
            {MSColumns::channel, ch},
            {MSColumns::polarization_product, pol}
          };
          output << "error at ("
                 << coords[0].second << ","
                 << coords[1].second << ","
                 << coords[2].second << ","
                 << coords[3].second << ","
                 << coords[4].second << "); "
                 << "value: ("
                 << value_map[coords[0].first] << ","
                 << value_map[coords[1].first] << ","
                 << value_map[coords[2].first] << ","
                 << value_map[coords[3].first] << ","
                 << value_map[coords[4].first] << ")"
                 << endl;
        }
        coords.pop_back();
      }
  }
  return result;
}

bool
cb(const MSArray& array, ostringstream& output) {

  if (!array.buffer())
    return false;
  auto indexes = array.blocks();

  output << "next buffer..." << endl;
  for (auto& seq : indexes) {
    output << mscol_nickname(seq.m_axis) << ": [";
    const char *sep = "";
    for (auto& block : seq.m_blocks) {
      output << sep << "(" << block.m_index << "," << block.m_length << ")";
      sep = ", ";
    }
    output << "]" << endl;
  }

  size_t n = 0;
  vector<pair<MSColumns, size_t> > coords;
  bool result =
    checkit(
      array.buffer().value(),
      n,
      coords,
      begin(indexes),
      end(indexes),
      output);
  if (result)
    output << "no errors" << endl;
  return result;
}

void
writeit(
  ostream& f,
  vector<ColumnAxisBase<MSColumns> >::const_iterator axis,
  vector<ColumnAxisBase<MSColumns> >::const_iterator end_axis,
  unordered_map<MSColumns, size_t>& index) {

  auto next_axis = axis;
  ++next_axis;
  if (next_axis != end_axis) {
    for (size_t i = 0; i < axis->length(); ++i) {
      index[axis->id()] = i;
      writeit(f, next_axis, end_axis, index);
    }
  } else {
    for (size_t i = 0; i < axis->length(); ++i) {
      index[axis->id()] = i;
      complex<float> vis;
      encode_vis(
        vis,
        index[MSColumns::time],
        index[MSColumns::spectral_window],
        index[MSColumns::baseline],
        index[MSColumns::channel],
        index[MSColumns::polarization_product]);
      f.write(reinterpret_cast<char *>(&vis), sizeof(vis));
    }
  }
}

void
write_file(
  const char *path,
  const vector<ColumnAxisBase<MSColumns> >& shape) {

  ofstream f(path, ofstream::out | ofstream::trunc | ofstream::binary);

  unordered_map<MSColumns, size_t> index;
  writeit(f, begin(shape), end(shape), index);
  f.close();
}

string
colnames(const vector<MSColumns>& cols) {
  ostringstream result;
  result << "(";
  const char *sep = "";
  for (auto& c : cols) {
    result << sep << mscol_nickname(c);
    sep = ",";
  }
  result << ")";
  return result.str();
}

std::size_t
num_elements(std::size_t sz) {
  return sz / sizeof(std::complex<float>);
}

int
main(int argc, char* argv[]) {

  if (argc != 2) {
    cerr << "usage: " << argv[0] << " <file>" << endl;
    return -1;
  }

  MPI_Init(&argc, &argv);
  set_throw_exception_errhandler(MPI_COMM_WORLD);

  vector<ColumnAxisBase<MSColumns> > ms_shape {
    ColumnAxis<MSColumns, MSColumns::time>(ntim),
      ColumnAxis<MSColumns, MSColumns::spectral_window>(nspw),
      ColumnAxis<MSColumns, MSColumns::baseline>(nbal),
      ColumnAxis<MSColumns, MSColumns::channel>(nch),
      ColumnAxis<MSColumns, MSColumns::polarization_product>(npol)
      };

  vector<ColumnAxisBase<MSColumns> > ms_shape_u {
    ColumnAxis<MSColumns, MSColumns::time>(),
      ColumnAxis<MSColumns, MSColumns::spectral_window>(nspw),
      ColumnAxis<MSColumns, MSColumns::baseline>(nbal),
      ColumnAxis<MSColumns, MSColumns::channel>(nch),
      ColumnAxis<MSColumns, MSColumns::polarization_product>(npol)
      };

  size_t max_buffer_size =
    ntim * nspw * nbal * nch * npol * sizeof(complex<float>);

  vector<size_t> buffer_sizes = {
    max_buffer_size,
    max_buffer_size / ntim,
    max_buffer_size / nch,
    npol * nch * sizeof(complex<float>)
  };

  vector<vector<MSColumns> > traversal_orders {
    {MSColumns::time, MSColumns::spectral_window, MSColumns::baseline,
        MSColumns::channel, MSColumns::polarization_product},
    {MSColumns::spectral_window, MSColumns::time, MSColumns::baseline,
        MSColumns::channel, MSColumns::polarization_product},
    {MSColumns::channel, MSColumns::spectral_window,
        MSColumns::time, MSColumns::baseline, MSColumns::polarization_product},
    {MSColumns::polarization_product, MSColumns::spectral_window,
        MSColumns::time, MSColumns::baseline, MSColumns::channel}
  };

  // unordered_map<MSColumns, DataDistribution> pgrid;

  unordered_map<MSColumns, DataDistribution> pgrid = {
    {MSColumns::spectral_window, DataDistribution { 2, 1 } },
    {MSColumns::channel, DataDistribution { 2, 3 } }
  };

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (my_rank == 0)
    write_file(argv[1], ms_shape);
  MPI_Barrier(MPI_COMM_WORLD);

  std::array<bool,2> ms_order{ false, true };

  try {
    for (auto& mso : ms_order) {
      for (auto& to : traversal_orders) {
        for (auto& bs : buffer_sizes) {
          for (auto& mss : {ms_shape /*, ms_shape_u*/}) {
            if (mss[0].is_unbounded() && to[0] !=  mss[0].id())
              continue;
            ostringstream output;
            output << "========= traversal_order "
                   << colnames(to)
                   << "; buffer_size "
                   << num_elements(bs)
                   << (mso ? " (ms order)" : "")
                   << (mss[0].is_unbounded() ? " (unspecified ms length)" : "")
                   << " ========="
                   << endl;
            auto reader =
              Reader::begin(
                argv[1],
                "native",
                MPI_COMM_WORLD,
                MPI_INFO_NULL,
                mss,
                to,
                mso,
                pgrid,
                bs,
                false);
            while (reader != Reader::end()) {
              const MSArray& array = *reader;
              if (array.buffer()) {
                if (cb(array, output))
                  ++reader;
                else
                  reader.interrupt();
              }
              else {
                ++reader;
              }
            }

            int output_rank = 0;
            while (output_rank < world_size) {
              if (output_rank == my_rank) {
                if (world_size > 1)
                  cout << "*************** rank "
                       << my_rank
                       << " ***************"
                       << endl;
                cout << output.str();
              }
              ++output_rank;
              MPI_Barrier(MPI_COMM_WORLD);
            }
          }
        }
      }
    }
    if (my_rank == 0)
      remove(argv[1]);
  } catch (...) {
    if (my_rank == 0)
      remove(argv[1]);
    throw;
  }
  MPI_Finalize();

return 0;
}
