#include <cmath>
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
isnan(const complex<float>& x) {
  return isnan(x.real()) || isnan(x.imag());
}

bool
checkit(
  const complex<float>* buffer,
  size_t& n,
  vector<pair<MSColumns, size_t> >& coords,
  complex<float>* full_array,
  shared_ptr<ArrayIndexer<MSColumns> >& full_array_indexer,
  vector<IndexBlockSequence<MSColumns> >::const_iterator begin,
  vector<IndexBlockSequence<MSColumns> >::const_iterator end,
  bool& index_out_of_range,
  ostringstream& output)
  __attribute__((unused));

bool
checkit(
  const complex<float>* buffer,
  size_t& n,
  vector<pair<MSColumns, size_t> >& coords,
  complex<float>* full_array,
  shared_ptr<ArrayIndexer<MSColumns> >& full_array_indexer,
  vector<IndexBlockSequence<MSColumns> >::const_iterator begin,
  vector<IndexBlockSequence<MSColumns> >::const_iterator end,
  bool& index_out_of_range,
  ostringstream& output) {

  auto next = begin;
  ++next;
  bool result = true;
  if (next != end) {
    for (auto& b : begin->m_blocks)
      for (size_t i = 0; i < b.m_length; ++i) {
        coords.emplace_back(begin->m_axis, b.m_index + i);
        result =
          checkit(
            buffer,
            n,
            coords,
            full_array,
            full_array_indexer,
            next,
            end,
            index_out_of_range,
            output)
          && result;
        coords.pop_back();
      }
  } else {
    for (auto& b : begin->m_blocks)
      for (size_t i = 0; i < b.m_length; ++i) {
        coords.emplace_back(begin->m_axis, b.m_index + i);
        unordered_map<MSColumns, size_t> coords_map(
          std::begin(coords), std::end(coords));
        try {
          auto offset = full_array_indexer->offset_of(coords_map);
          if (offset)
            full_array[offset.value()] = buffer[n];
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
        } catch (const std::out_of_range& oor) {
          index_out_of_range = true;
        }
        coords.pop_back();
      }
  }
  return result;
}

bool
cb(
  const CxFltMSArray& array,
  complex<float>* full_array,
  shared_ptr<ArrayIndexer<MSColumns> >& full_array_indexer,
  bool& index_out_of_range,
  ostringstream& output) {

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
      full_array,
      full_array_indexer,
      begin(indexes),
      end(indexes),
      index_out_of_range,
      output);
  if (result)
    output << "no errors" << endl;

  return result;
}

void
merge_full_arrays(
  MPI_Win win,
  int rank,
  complex<float>* full_array,
  std::size_t array_length) {

  MPI_Group group;
  MPI_Win_get_group(win, &group);
  MPI_Group g0, grest;
  int r0 = 0;
  MPI_Group_incl(group, 1, &r0, &g0);
  MPI_Group_excl(group, 1, &r0, &grest);
  if (rank == 0) {
    MPI_Win_post(grest, MPI_MODE_NOPUT, win);
    MPI_Win_wait(win);
  } else {
    MPI_Win_start(g0, 0, win);
    std::size_t i = 0;
    while (i < array_length) {
      int count = 0;
      while (i < array_length && !isnan(full_array[i])) {
        ++count;
        ++i;
      }
      if (count > 0)
        MPI_Put(
          full_array + i - count,
          count,
          MPI_CXX_FLOAT_COMPLEX,
          0,
          i - count,
          count,
          MPI_CXX_FLOAT_COMPLEX,
          win);
      else
        ++i;
    }
    MPI_Win_complete(win);
  }
  MPI_Group_free(&g0);
  MPI_Group_free(&grest);
  MPI_Group_free(&group);
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

  size_t max_buffer_length =
    ntim * nspw * nbal * nch * npol;

  size_t max_buffer_size = max_buffer_length * sizeof(complex<float>);

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

  // unordered_map<MSColumns, GridDistribution> pgrid;

  unordered_map<MSColumns, GridDistribution> pgrid = {
    {MSColumns::spectral_window, GridDistributionFactory::cyclic(1, 2) },
    {MSColumns::channel, GridDistributionFactory::cyclic(3, 2) }
  };

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto full_array_idx =
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, ms_shape);
  complex<float>* full_array;
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "same_size", "true");
  MPI_Win full_array_win;
  MPI_Win_allocate(
    max_buffer_size,
    sizeof(complex<float>),
    info,
    MPI_COMM_WORLD,
    &full_array,
    &full_array_win);
  MPI_Info_free(&info);

  if (my_rank == 0)
    write_file(argv[1], ms_shape);
  MPI_Barrier(MPI_COMM_WORLD);

  std::array<bool,2> ms_order{ false, true };

  try {
    for (auto& mso : ms_order) {
      for (auto& to : traversal_orders) {
        for (auto& bs : buffer_sizes) {
          for (auto& mss : {ms_shape, ms_shape_u}) {
            if (mss[0].is_indeterminate() && to[0] !=  mss[0].id())
              continue;

            for (std::size_t i = 0; i < max_buffer_length; ++i)
              full_array[i] = complex{NAN, NAN};

            ostringstream output;
            output << "========= traversal_order "
                   << colnames(to)
                   << "; buffer_size "
                   << num_elements(bs)
                   << (mso ? " (ms order)" : "")
                   << (mss[0].is_indeterminate()
                       ? " (unspecified ms length)"
                       : "")
                   << " ========="
                   << endl;
            auto reader =
              CxFltReader::begin(
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

            bool index_oor = false;
            while (reader != CxFltReader::end()) {
              const CxFltMSArray& array = *reader;
              if (array.buffer()) {
                if (cb(array, full_array, full_array_idx, index_oor, output))
                  ++reader;
                else
                  reader.interrupt();
              } else {
                ++reader;
              }
            }

            merge_full_arrays(
              full_array_win,
              my_rank,
              full_array,
              max_buffer_length);

            int output_rank = 0;
            while (output_rank < world_size) {
              if (output_rank == my_rank) {
                if (world_size > 1)
                  cout << "*************** rank "
                       << my_rank
                       << " ***************"
                       << endl;
                if (index_oor)
                  output << "---- error: out of range array indexes ---- "
                         << endl;
                cout << output.str();
              }
              ++output_rank;
              MPI_Barrier(MPI_COMM_WORLD);
            }

            if (my_rank == 0) {
              std::size_t num_missing = 0;
              for (std::size_t i = 0; i < max_buffer_length; ++i)
                if (isnan(full_array[i]))
                  ++num_missing;
              cout << "++++++++ ";
              if (num_missing == 0)
                cout << "no";
              else
                cout << num_missing;
              cout << " missing elements ++++++++" << endl;
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
  MPI_Win_free(&full_array_win);
  MPI_Finalize();

  return 0;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
