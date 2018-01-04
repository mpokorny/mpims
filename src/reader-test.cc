/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <complex>
#include <fstream>
#include <iostream>
#include <memory>

#include <mpi.h>

#include <mpims.h>
#include <ColumnAxis.h>
#include <Reader.h>

using namespace mpims;

constexpr auto ntim = 4;
constexpr auto nbits_tim = 2;
constexpr auto nspw = 2;
constexpr auto nbits_spw = 1;
constexpr auto nbal = 6;
constexpr auto nbits_bal = 3;
constexpr auto nch = 8;
constexpr auto nbits_ch = 3;
constexpr auto npol = 2;
constexpr auto nbits_pol = 1;

void
encode_vis(
  std::complex<float>& vis,
  std::size_t tim,
  std::size_t spw,
  std::size_t bal,
  std::size_t ch,
  std::size_t pol) {

  vis.real(static_cast<float>((((bal << nbits_spw) | spw) << nbits_tim) | tim));
  vis.imag(static_cast<float>((ch << nbits_pol) | pol));
}

void
decode_vis(
  const std::complex<float>& vis,
  std::size_t& tim,
  std::size_t& spw,
  std::size_t& bal,
  std::size_t& ch,
  std::size_t& pol) {

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

void
cb(
  const std::vector<IndexBlockSequence<MSColumns> >& indexes,
  std::shared_ptr<std::complex<float> >& buffer) {

  for (auto& seq : indexes) {
    std::cout << mscol_nickname(seq.m_axis) << ": [";
    const char *sep = "";
    for (auto& block : seq.m_blocks) {
      std::cout << sep << "(" << block.m_index << "," << block.m_length << ")";
      sep = ", ";
    }
    std::cout << "]" << std::endl;
  }
}

void
writeit(
  std::ofstream& f,
  std::vector<ColumnAxisBase<MSColumns> >::const_iterator axis,
  std::vector<ColumnAxisBase<MSColumns> >::const_iterator end_axis,
  std::unordered_map<MSColumns, std::size_t>& index) {

  auto next_axis = axis;
  ++next_axis;
  if (next_axis != end_axis) {
    for (std::size_t i = 0; i < axis->length(); ++i) {
      index[axis->id()] = i;
      writeit(f, next_axis, end_axis, index);
    }
  } else {
    for (std::size_t i = 0; i < axis->length(); ++i) {
      index[axis->id()] = i;
      std::complex<float> vis;
      encode_vis(
        vis,
        index[MSColumns::time],
        index[MSColumns::spectral_window],
        index[MSColumns::baseline],
        index[MSColumns::channel],
        index[MSColumns::polarization_product]);
      f << vis;
    }
  }
}

void
write_file(
  const char *path,
  const std::vector<ColumnAxisBase<MSColumns> >& shape) {

  std::ofstream f(
    path,
    std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

  std::unordered_map<MSColumns, std::size_t> index;
  writeit(f, std::begin(shape), std::end(shape), index);
  f.close();
}

int
main(int argc, char* argv[]) {

  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <file>" << std::endl;
    return -1;
  }

  ::MPI_Init(&argc, &argv);
  ::MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  std::vector<ColumnAxisBase<MSColumns> > ms_shape {
    ColumnAxis<MSColumns, MSColumns::time>(ntim),
      ColumnAxis<MSColumns, MSColumns::spectral_window>(nspw),
      ColumnAxis<MSColumns, MSColumns::baseline>(nbal),
      ColumnAxis<MSColumns, MSColumns::channel>(nch),
      ColumnAxis<MSColumns, MSColumns::polarization_product>(npol)
  };
  std::vector<MSColumns> traversal_order {
    MSColumns::time, MSColumns::spectral_window, MSColumns::baseline,
      MSColumns::channel, MSColumns::polarization_product
  };

  write_file(argv[1], ms_shape);

  std::unordered_map<MSColumns, ProcessDistribution> pgrid;
  std::size_t buffer_size =
    ntim * nspw * nbal * nch * npol * sizeof(std::complex<float>);

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
