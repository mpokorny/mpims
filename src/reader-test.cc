/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <complex>
#include <fstream>
#include <iostream>
#include <memory>
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
constexpr auto nspw = 2;
constexpr auto nbits_spw = 1;
constexpr auto nbal = 3 /*6*/;
constexpr auto nbits_bal = 3;
constexpr auto nch = 2 /*8*/;
constexpr auto nbits_ch = 3;
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
  const shared_ptr<complex<float> >& buffer,
  size_t& n,
  vector<pair<MSColumns, size_t> >& coords,
  vector<IndexBlockSequence<MSColumns> >::const_iterator begin,
  vector<IndexBlockSequence<MSColumns> >::const_iterator end)
  __attribute__((unused));

bool
checkit(
  const shared_ptr<complex<float> >& buffer,
  size_t& n,
  vector<pair<MSColumns, size_t> >& coords,
  vector<IndexBlockSequence<MSColumns> >::const_iterator begin,
  vector<IndexBlockSequence<MSColumns> >::const_iterator end) {

  auto next = begin;
  ++next;
  bool result = true;
  if (next != end) {
    for (auto& b : begin->m_blocks)
      for (size_t i = 0; i < b.m_length; ++i) {
        coords.emplace_back(begin->m_axis, b.m_index + i);
        result = checkit(buffer, n, coords, next, end) && result;
        coords.pop_back();
      }
  } else {
    for (auto& b : begin->m_blocks)
      for (size_t i = 0; i < b.m_length; ++i) {
        coords.emplace_back(begin->m_axis, b.m_index + i);
        unordered_map<MSColumns, size_t> coords_map(
          std::begin(coords), std::end(coords));
        size_t tim, spw, bal, ch, pol;
        decode_vis(buffer.get()[n++], tim, spw, bal, ch, pol);
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
          std::cout << "error at ("
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
                    << std::endl;
        }
        coords.pop_back();
      }
  }
  return result;
}

void
cb(
  const vector<IndexBlockSequence<MSColumns> >& indexes,
  shared_ptr<complex<float> >& buffer) {

  std::cout << "next buffer..." << std::endl;
  for (auto& seq : indexes) {
    cout << mscol_nickname(seq.m_axis) << ": [";
    const char *sep = "";
    for (auto& block : seq.m_blocks) {
      cout << sep << "(" << block.m_index << "," << block.m_length << ")";
      sep = ", ";
    }
    cout << "]" << endl;
  }

  size_t n = 0;
  vector<pair<MSColumns, size_t> > coords;
  if (checkit(buffer, n, coords, begin(indexes), end(indexes)))
    std::cout << "no errors" << std::endl;
}

void
writeit(
  ofstream& f,
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

int
main(int argc, char* argv[]) {

  if (argc != 2) {
    cerr << "usage: " << argv[0] << " <file>" << endl;
    return -1;
  }

  ::MPI_Init(&argc, &argv);
  ::MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  vector<ColumnAxisBase<MSColumns> > ms_shape {
    ColumnAxis<MSColumns, MSColumns::time>(ntim),
      ColumnAxis<MSColumns, MSColumns::spectral_window>(nspw),
      ColumnAxis<MSColumns, MSColumns::baseline>(nbal),
      ColumnAxis<MSColumns, MSColumns::channel>(nch),
      ColumnAxis<MSColumns, MSColumns::polarization_product>(npol)
  };
  vector<MSColumns> traversal_order {
    MSColumns::spectral_window, MSColumns::time, MSColumns::baseline,
      MSColumns::channel, MSColumns::polarization_product};
  // vector<MSColumns> traversal_order {
  //   MSColumns::time, MSColumns::spectral_window, MSColumns::baseline,
  //   MSColumns::channel, MSColumns::polarization_product};

  write_file(argv[1], ms_shape);

  unordered_map<MSColumns, ProcessDistribution> pgrid;
  size_t buffer_size =
    ntim * nspw * nbal * nch * npol * sizeof(complex<float>) / nspw;

  Reader reader(
    argv[1],
    MPI_COMM_WORLD,
    MPI_INFO_NULL,
    ms_shape,
    traversal_order,
    pgrid,
    buffer_size);
  reader.iterate(cb);
  reader.finalize();
  ::MPI_Finalize();

  return 0;
}
