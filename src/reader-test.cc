#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
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
  unordered_map<MSColumns, set<size_t> >& indexes,
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
  unordered_map<MSColumns, set<size_t> >& indexes,
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
            indexes,
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
        for_each(
          std::begin(coords_map),
          std::end(coords_map),
          [&](auto& ci) {
            MSColumns col;
            size_t idx;
            tie(col, idx) = ci;
            indexes[col].insert(idx);
          });
        try {
          auto offset = full_array_indexer->offset_of(coords_map);
          if (offset)
            full_array[offset.value()] = (isnan(buffer[n]) ? 0.0 : buffer[n]);
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
        } catch (const out_of_range& oor) {
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
  unordered_map<MSColumns, set<size_t> >& indexes,
  ostringstream& output) {

  if (!array.buffer())
    return false;
  auto array_indexes = array.blocks();

  output << "next buffer..." << endl;
  for (auto& seq : array_indexes) {
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
      begin(array_indexes),
      end(array_indexes),
      index_out_of_range,
      indexes,
      output);

  return result;
}

void
merge_full_arrays(
  MPI_Win win,
  int rank,
  complex<float>* full_array,
  size_t array_length) {

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
    size_t i = 0;
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

vector<vector<size_t> >
gather_indexes(int num_ranks, const set<size_t>& indexes) {
  vector<int> all_num_indexes(num_ranks);
  vector<size_t> is(std::begin(indexes), std::end(indexes));
  int num_indexes = is.size();
  MPI_Gather(
    &num_indexes,
    1,
    MPI_INT,
    all_num_indexes.data(),
    1,
    MPI_INT,
    0,
    MPI_COMM_WORLD);
  int total_num_indexes =
    accumulate(std::begin(all_num_indexes), std::end(all_num_indexes), 0);
  vector<size_t> all_indexes(total_num_indexes);
  vector<int> displacements;
  displacements.emplace_back(0);
  partial_sum(
    std::begin(all_num_indexes),
    std::end(all_num_indexes),
    std::back_inserter(displacements));
  MPI_Gatherv(
    is.data(),
    num_indexes,
    MPIMS_SIZE_T,
    all_indexes.data(),
    all_num_indexes.data(),
    displacements.data(),
    MPIMS_SIZE_T,
    0,
    MPI_COMM_WORLD);
  vector<vector<size_t> > result;
  for (int r = 0; r < num_ranks; ++r) {
    vector<size_t> ris;
    for (int i = 0; i < all_num_indexes[r]; ++i)
      ris.push_back(all_indexes[i + displacements[r]]);
    result.emplace_back(std::move(ris));
  }
  return result;
}

void
fill_grid(
  const unordered_map<MSColumns, shared_ptr<const DataDistribution> >& dists,
  vector<ColumnAxisBase<MSColumns> >::const_iterator begin,
  vector<ColumnAxisBase<MSColumns> >::const_iterator end,
  vector<pair<MSColumns, size_t> >& grid_coords,
  shared_ptr<ArrayIndexer<MSColumns> >& grid_indexer,
  vector<std::map<MSColumns, vector<size_t> > >& grid_distribution) {

  auto next = begin;
  ++next;
  if (next != end) {
    auto len = begin->length().value();
    auto col = begin->id();
    for (size_t i = 0; i < len; ++i) {
      grid_coords.emplace_back(col, i);
      fill_grid(
        dists,
        next,
        end,
        grid_coords,
        grid_indexer,
        grid_distribution);
      grid_coords.pop_back();
    }
  } else {
    auto len = begin->length().value();
    auto col = begin->id();
    for (size_t i = 0; i < len; ++i) {
      grid_coords.emplace_back(col, i);
      unordered_map<MSColumns, size_t> coords_map(
        std::begin(grid_coords), std::end(grid_coords));
      std::map<MSColumns, vector<size_t> > grid_indexes;
      for_each(
        std::begin(coords_map),
        std::end(coords_map),
        [&grid_indexes, &dists](auto& ci) {
          MSColumns col;
          size_t idx;
          std::tie(col, idx) = ci;
          grid_indexes[col] = dists.at(col)->begin(idx)->take_all();
        });
      grid_distribution[grid_indexer->offset_of(coords_map).value()] =
        grid_indexes;
      grid_coords.pop_back();
    }
  }
}

bool
check_grid_distribution(
  const unordered_map<MSColumns, GridDistribution>& pgrid,
  const vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const unordered_map<MSColumns, set<size_t> >& indexes) {

  // gather to rank 0 the indexes received by each rank, by column id
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  unordered_map<MSColumns, vector<vector<size_t> > > all_indexes;
  for_each(
    begin(ms_shape),
    end(ms_shape),
    [&num_ranks, &indexes, &all_indexes](auto& ax) {
      all_indexes[ax.id()] = gather_indexes(num_ranks, indexes.at(ax.id()));
    });

  // transpose all_indexes to create a vector indexed by rank of received
  // indexes for each column id; use a map rather than unsorted_map here to
  // enable the immediately following sort() call
  vector<std::map<MSColumns, vector<size_t> > > rank_indexes(num_ranks);
  for_each(
    begin(all_indexes),
    end(all_indexes),
    [&rank_indexes, &num_ranks](auto& cis) {
      MSColumns col;
      vector<vector<size_t> > is;
      std::tie(col, is) = cis;
      for (int r = 0; r < num_ranks; ++r)
        rank_indexes[r][col] = is[r];
    });
  // since we don't really care about rank values (which is an implementation
  // detail), we will sort rank_indexes for simpler comparison with the expected
  // index sets
  sort(begin(rank_indexes), end(rank_indexes));

  // create DataDistributions for the process grid, using the ms axis lengths in
  // ms_shape, and creating "unpartitioned" distributions for all axes not in
  // pgrid
  unordered_map<MSColumns, shared_ptr<const DataDistribution> > dists;
  for_each(
    begin(ms_shape),
    end(ms_shape),
    [&pgrid, &dists](auto& ax) {
      if (pgrid.count(ax.id()) > 0)
        dists[ax.id()] = pgrid.at(ax.id())(ax.length());
      else
        dists[ax.id()] = GridDistributionFactory::unpartitioned()(ax.length());
    });

  // create an array axis specification (or shape) for the process grid
  vector<ColumnAxisBase<MSColumns> > grid_shape;
  transform(
    begin(ms_shape),
    end(ms_shape),
    back_inserter(grid_shape),
    [&dists](auto& ax) {
      return ColumnAxisBase<MSColumns>(
        static_cast<unsigned>(ax.id()), dists[ax.id()]->order());
    });
  // compute size of process grid
  auto grid_size =
    accumulate(
      begin(grid_shape),
      end(grid_shape),
      1,
      [](auto& sz, auto& ax){
        return sz * ax.length().value();
      });
  // create an array for the process grid, in which we will store the expected
  // index sets for each process in the grid; again, use map rather than
  // unsorted_map since we intend to sort the vector
  auto grid_distribution =
    vector<std::map<MSColumns, vector<size_t> > >(grid_size);
  // we can use an ArrayIndexer to index the grid_distribution array
  auto grid_indexer =
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, grid_shape);
  // fill the grid_distribution array with the expected index sets
  vector<pair<MSColumns, size_t> > grid_coords;
  fill_grid(
    dists,
    begin(grid_shape),
    end(grid_shape),
    grid_coords,
    grid_indexer,
    grid_distribution);
  // again, we don't care about the ordering of elements in grid_distribution,
  // we will sort that array for easier comparison with rank_indexes
  sort(begin(grid_distribution), end(grid_distribution));

  return grid_distribution == rank_indexes;
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

size_t
num_elements(size_t sz) {
  return sz / sizeof(complex<float>);
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

  size_t max_buffer_length = ntim * nspw * nbal * nch * npol;

  size_t max_buffer_size = max_buffer_length * sizeof(complex<float>);

  vector<size_t> buffer_sizes {
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

  unordered_map<MSColumns, GridDistribution> pgrid1;

  unordered_map<MSColumns, GridDistribution> pgrid2 {
    {MSColumns::spectral_window, GridDistributionFactory::block_sequence(
        {{{0, 1}, {3, 1}, {4, 0}},
         {{1, 2}, {4, 0}}})}
  };

  unordered_map<MSColumns, GridDistribution> pgrid4 {
    {MSColumns::spectral_window, GridDistributionFactory::cyclic(1, 2) },
    {MSColumns::channel, GridDistributionFactory::cyclic(3, 2) }
  };

  unordered_map<int, unordered_map<MSColumns, GridDistribution> > pgrids {
    {1, pgrid1},
    {2, pgrid2},
    {4, pgrid4}
  };

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int grid_size = world_size;
  auto pgrid = pgrids.find(grid_size);
  while (pgrid == end(pgrids))
    pgrid = pgrids.find(--grid_size);

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

  array<bool,2> ms_order{ false, true };

  for (auto& mso : ms_order) {
    for (auto& to : traversal_orders) {
      for (auto& bs : buffer_sizes) {
        for (auto& mss : {ms_shape, ms_shape_u}) {
          if (mss[0].is_indeterminate() && to[0] !=  mss[0].id())
            continue;

          for (size_t i = 0; i < max_buffer_length; ++i)
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
              pgrid->second,
              bs,
              false);

          bool index_oor = false;
          unordered_map<MSColumns, set<size_t> > indexes;
          while (reader != CxFltReader::end()) {
            const CxFltMSArray& array = *reader;
            if (array.buffer()) {
              if (cb(
                    array,
                    full_array,
                    full_array_idx,
                    index_oor,
                    indexes,
                    output))
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

          bool distribution_ok =
            check_grid_distribution(pgrid->second, ms_shape, indexes);

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
            size_t num_missing = 0;
            for (size_t i = 0; i < max_buffer_length; ++i)
              if (isnan(full_array[i]))
                ++num_missing;
            if (num_missing > 0)
              cout << "++++ error: "
                   << num_missing
                   << " missing elements ++++" << endl;
            if (!distribution_ok)
              cout << "#### error: distribution failure ####" << endl;
          }
        }
      }
    }
  }
  if (my_rank == 0)
    remove(argv[1]);

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
