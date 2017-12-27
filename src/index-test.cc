/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <iostream>

#include <ArrayIndexer.h>

using namespace std;
using namespace mpims;

enum class MSColumns {
  time,
  spectral_window,
  baseline,
  channel,
  polarization_product
};

int
main(int argc, char* argv[]) {

  constexpr size_t time_len = 3;
  constexpr size_t spw_len = 2;
  constexpr size_t bal_len = 3;
  constexpr size_t ch_len = 4;
  constexpr size_t pol_len = 2;

  std::vector<ColumnAxisBase<MSColumns> > shape {
    ColumnAxis<MSColumns, MSColumns::time>(time_len),
      ColumnAxis<MSColumns, MSColumns::spectral_window>(spw_len),
      ColumnAxis<MSColumns, MSColumns::baseline>(bal_len),
      ColumnAxis<MSColumns, MSColumns::channel>(ch_len),
      ColumnAxis<MSColumns, MSColumns::polarization_product>(pol_len)};

  {
    auto indexer = ArrayIndexer<MSColumns>::index_of(ArrayOrder::row_major, shape);
    for (unsigned t = 0; t < time_len; ++t) {
      std::unordered_map<MSColumns, size_t> index;
      index[MSColumns::time] = t;
      for (unsigned spw = 0; spw < spw_len; ++spw) {
        index[MSColumns::spectral_window] = spw;
        for (unsigned bal = 0; bal < bal_len; ++bal) {
          index[MSColumns::baseline] = bal;
          for (unsigned ch = 0; ch < ch_len; ++ch) {
            index[MSColumns::channel] = ch;
            for (unsigned pol = 0; pol < pol_len; ++pol) {
              index[MSColumns::polarization_product] = pol;
              std::cout << indexer->offset_of(index) << std::endl;
            }
          }
        }
      }
    }
  }
  {
    auto indexer = ArrayIndexer<MSColumns>::index_of(ArrayOrder::column_major, shape);
    for (unsigned pol = 0; pol < pol_len; ++pol) {
      std::unordered_map<MSColumns, size_t> index;
      index[MSColumns::polarization_product] = pol;
      for (unsigned ch = 0; ch < ch_len; ++ch) {
        index[MSColumns::channel] = ch;
        for (unsigned bal = 0; bal < bal_len; ++bal) {
          index[MSColumns::baseline] = bal;
          for (unsigned spw = 0; spw < spw_len; ++spw) {
            index[MSColumns::spectral_window] = spw;
            for (unsigned t = 0; t < time_len; ++t) {
              index[MSColumns::time] = t;
              std::cout << indexer->offset_of(index) << std::endl;
            }
          }
        }
      }
    }
  }
}
