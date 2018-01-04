/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <iostream>

#include <ArrayIndexer.h>
#include <MSColumns.h>

using namespace std;
using namespace mpims;

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
    std::cout << "*** row major order and traversal ***" << std::endl;
    auto indexer = ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, shape);
    for (unsigned t = 0; t < time_len; ++t) {
      ArrayIndexer<MSColumns>::index index;
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
    std::cout << "---------------" << std::endl;

    std::cout << "*** row major order, slice of second spw ***" << std::endl;
    ArrayIndexer<MSColumns>::index slice;
    slice[MSColumns::spectral_window] = 1;
    auto spw1 = indexer->slice(slice);
    for (unsigned t = 0; t < time_len; ++t) {
      ArrayIndexer<MSColumns>::index index;
      index[MSColumns::time] = t;
      for (unsigned bal = 0; bal < bal_len; ++bal) {
        index[MSColumns::baseline] = bal;
        for (unsigned ch = 0; ch < ch_len; ++ch) {
          index[MSColumns::channel] = ch;
          for (unsigned pol = 0; pol < pol_len; ++pol) {
            index[MSColumns::polarization_product] = pol;
            std::cout << spw1->offset_of(index) << std::endl;
          }
        }
      }
    }
    std::cout << "---------------" << std::endl;

    std::cout << "*** row major order, slice of second spw & fourth channel ***" << std::endl;
    ArrayIndexer<MSColumns>::index slice1;
    slice1[MSColumns::channel] = 3;
    auto ch3 = spw1->slice(slice1);
    for (unsigned t = 0; t < time_len; ++t) {
      ArrayIndexer<MSColumns>::index index;
      index[MSColumns::time] = t;
      for (unsigned bal = 0; bal < bal_len; ++bal) {
        index[MSColumns::baseline] = bal;
        for (unsigned pol = 0; pol < pol_len; ++pol) {
          index[MSColumns::polarization_product] = pol;
          std::cout << ch3->offset_of(index) << std::endl;
        }
      }
    }
    std::cout << "---------------" << std::endl;

    std::cout << "*** previous slice, attempt all channels" << std::endl;
    {
      ArrayIndexer<MSColumns>::index index;
      index[MSColumns::time] = 0;
      index[MSColumns::baseline] = 0;
      index[MSColumns::polarization_product] = 0;
      for (unsigned ch = 0; ch < ch_len; ++ch) {
        index[MSColumns::channel] = ch;
        std::cout << ch3 ->offset_of(index) << std::endl;
      }
    }
    std::cout << "---------------" << std::endl;
  }
  {
    std::cout << "*** column major order and traversal ***" << std::endl;
    auto indexer = ArrayIndexer<MSColumns>::of(ArrayOrder::column_major, shape);
    for (unsigned pol = 0; pol < pol_len; ++pol) {
      ArrayIndexer<MSColumns>::index index;
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
