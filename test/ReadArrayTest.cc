#include <cassert>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <mpims.h>
#include <ColumnAxis.h>
#include <Reader.h>

#include "gtest/gtest.h"

using namespace std;
using namespace mpims;

// helper functions for test assertions
bool
all_true(MPI_Comm comm, bool p) {
  MPI_Allreduce(MPI_IN_PLACE, &p, 1, MPI_CXX_BOOL, MPI_LAND, comm);
  return p;
}

bool
all_false(MPI_Comm comm, bool p) {
  MPI_Allreduce(MPI_IN_PLACE, &p, 1, MPI_CXX_BOOL, MPI_LOR, comm);
  return !p;
}

// shape of test MS data column
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

// encode column index as visibility value
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

// decode visibility value to column index
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

// isnan() for complex values
bool
isnan(const complex<float>& x) {
  return isnan(x.real()) || isnan(x.imag());
}

// check encoded visibility values in buffer
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
  unordered_map<MSColumns, set<size_t> >& indexes) {

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
            indexes)
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
          }
        } catch (const out_of_range& oor) {
          index_out_of_range = true;
        }
        coords.pop_back();
      }
  }
  return result;
}

// entry point for checkit()
bool
cb(
  const CxFltMSArray& array,
  complex<float>* full_array,
  shared_ptr<ArrayIndexer<MSColumns> >& full_array_indexer,
  bool& index_out_of_range,
  unordered_map<MSColumns, set<size_t> >& indexes) {

  if (!array.buffer())
    return false;
  auto array_indexes = array.blocks();

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
      indexes);

  return result;
}

// merge arrays from all ranks
//
// perhaps we could get rid of this and write directly into an MPI_Win as
// buffers are returned from Reader (in checkit())
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

// get column indexes read by every rank
vector<vector<size_t> >
gather_indexes(MPI_Comm comm, int num_ranks, const set<size_t>& indexes) {

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
    comm);
  int total_num_indexes =
    accumulate(std::begin(all_num_indexes), std::end(all_num_indexes), 0);
  vector<size_t> all_indexes(total_num_indexes);
  vector<int> displacements;
  displacements.emplace_back(0);
  partial_sum(
    begin(all_num_indexes),
    end(all_num_indexes),
    back_inserter(displacements));
  MPI_Gatherv(
    is.data(),
    num_indexes,
    MPIMS_SIZE_T,
    all_indexes.data(),
    all_num_indexes.data(),
    displacements.data(),
    MPIMS_SIZE_T,
    0,
    comm);
  vector<vector<size_t> > result;
  for (int r = 0; r < num_ranks; ++r) {
    vector<size_t> ris;
    for (int i = 0; i < all_num_indexes[r]; ++i)
      ris.push_back(all_indexes[i + displacements[r]]);
    result.emplace_back(std::move(ris));
  }
  return result;
}

// fill a vector with expected data column indexes according to a given
// DataDistribution
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

// check that data column indexes read by every rank are as expected
bool
check_grid_distribution(
  MPI_Comm comm,
  const unordered_map<MSColumns, GridDistribution>& pgrid,
  const vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const unordered_map<MSColumns, set<size_t> >& indexes) {

  // gather to rank 0 the indexes received by each rank, by column id
  int num_ranks;
  MPI_Comm_size(comm, &num_ranks);
  unordered_map<MSColumns, vector<vector<size_t> > > all_indexes;
  for_each(
    begin(ms_shape),
    end(ms_shape),
    [comm, &num_ranks, &indexes, &all_indexes](auto& ax) {
      all_indexes[ax.id()] =
        gather_indexes(comm, num_ranks, indexes.at(ax.id()));
    });

  // transpose all_indexes to create a vector (indexed by rank) of received
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

// write values to MS data column file
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

vector<ColumnAxisBase<MSColumns> >
complete_shape(const vector<ColumnAxisBase<MSColumns> >& shape) {

  // fix up outer axis length for indeterminate MS data column shapes
  vector<ColumnAxisBase<MSColumns> > result = shape;
  ColumnAxisBase<MSColumns>& outer = result[0];
  if (outer.is_indeterminate()) {
    switch (outer.id()) {
    case MSColumns::time:
      outer =
        ColumnAxisBase<MSColumns>(
          static_cast<unsigned>(MSColumns::time),
          ntim);
      break;
    case MSColumns::spectral_window:
      outer =
        ColumnAxisBase<MSColumns>(
          static_cast<unsigned>(MSColumns::spectral_window),
          nspw);
      break;
    case MSColumns::baseline:
      outer =
        ColumnAxisBase<MSColumns>(
          static_cast<unsigned>(MSColumns::baseline),
          nbal);
      break;
    case MSColumns::channel:
      outer =
        ColumnAxisBase<MSColumns>(
          static_cast<unsigned>(MSColumns::channel),
          nch);
      break;
    case MSColumns::polarization_product:
      outer =
        ColumnAxisBase<MSColumns>(
          static_cast<unsigned>(MSColumns::polarization_product),
          npol);
      break;
    case MSColumns::complex:
      outer =
        ColumnAxisBase<MSColumns>(
          static_cast<unsigned>(MSColumns::complex),
          2);
      break;
    default:
      outer = ColumnAxisBase<MSColumns>(static_cast<unsigned>(outer.id()), 1);
      break;
    }
  }
  return result;
}

// write a test MS data column file
std::experimental::filesystem::path
write_file(
  MPI_Comm comm,
  const std::experimental::filesystem::path& testdir,
  const vector<ColumnAxisBase<MSColumns> >& shape) {

  int rank;
  MPI_Comm_rank(comm, &rank);

  string filename;
  if (rank == 0) {
    filename = (testdir / "XXXXXX").string();
    int fd = mkstemp(filename.data());
    close(fd);
    ofstream f(filename, ofstream::out | ofstream::binary);
    unordered_map<MSColumns, size_t> index;
    writeit(f, begin(shape), end(shape), index);
    f.close();

    size_t fnsz = filename.size() + 1;
    MPI_Bcast(&fnsz, 1, MPIMS_SIZE_T, 0, comm);
    MPI_Bcast(const_cast<char*>(filename.c_str()), fnsz, MPI_CHAR, 0, comm);
  } else {
    size_t fnsz;
    MPI_Bcast(&fnsz, 1, MPIMS_SIZE_T, 0, comm);
    auto c = make_unique<char []>(fnsz);
    MPI_Bcast(c.get(), fnsz, MPI_CHAR, 0, comm);
    filename = c.get();
  }
  return filename;
}

// main test routine for distributed parallel read of MS data column file
void
read_array_test(
  const char *testdir,
  MPI_Comm comm,
  bool ms_order,
  const vector<MSColumns>& traversal_order,
  size_t buffer_size,
  const vector<ColumnAxisBase<MSColumns> >& ms_shape,
  size_t max_ms_length,
  const unordered_map<MSColumns, GridDistribution>& pgrid,
  bool readahead) {

  // maintain an array, initially filled by NAN values, to track the data column
  // indexes that have been read
  complex<float>* full_array;
  size_t full_size = max_ms_length * sizeof(complex<float>);
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "same_size", "true");
  MPI_Win full_array_win;
  MPI_Win_allocate(
    full_size,
    sizeof(complex<float>),
    info,
    comm,
    &full_array,
    &full_array_win);
  MPI_Info_free(&info);
  for (size_t i = 0; i < max_ms_length; ++i)
    full_array[i] = complex{NAN, NAN};
  auto full_array_idx =
    ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, ms_shape);

  // write the test MS data column file
  auto complete_ms_shape = complete_shape(ms_shape);
  auto filepath = write_file(comm, testdir, complete_ms_shape);

  // create the Reader instance
  auto reader =
    CxFltReader::begin(
      filepath.string(),
      "native",
      comm,
      MPI_INFO_NULL,
      ms_shape,
      traversal_order,
      ms_order,
      pgrid,
      buffer_size,
      readahead);

  // iterate the Reader, calling cb() on every returned MSArray
  bool index_oor = false;
  unordered_map<MSColumns, set<size_t> > indexes;
  bool cb_result = true;
  while (reader != CxFltReader::end()) {
    const CxFltMSArray& array = *reader;
    if (array.buffer()) {
      cb_result = cb(array, full_array, full_array_idx, index_oor, indexes);
      if (cb_result)
        ++reader;
      else
        reader.interrupt();
    } else {
      ++reader;
    }
  }
  // check that no call to cb() returned 'false'
  EXPECT_TRUE(all_true(comm, cb_result));
  // check that no data column indexes received by Reader were out of range
  EXPECT_TRUE(all_false(comm, index_oor));

  // merge the full_array arrays, then check that no NAN values are present
  int rank;
  MPI_Comm_rank(comm, &rank);
  merge_full_arrays(full_array_win, rank, full_array, max_ms_length);
  bool all_present = true;
  for (size_t i = 0; all_present && i < max_ms_length; ++i)
    all_present = !isnan(full_array[i]);
  EXPECT_TRUE(all_true(comm, rank > 0 || all_present));

  // check that the data column indexes received by every rank conform to the
  // GridDistribution used
  bool distribution_ok =
    check_grid_distribution(comm, pgrid, complete_ms_shape, indexes);
  EXPECT_TRUE(all_true(comm, rank > 0 || distribution_ok));

  MPI_Win_free(&full_array_win);
  if (rank == 0)
    remove(filepath);
}

// test parameters
struct read_array_test_params {
  bool ms_order;
  const vector<MSColumns>& traversal_order;
  size_t buffer_length;
  const vector<ColumnAxisBase<MSColumns> >& ms_shape;
  size_t max_ms_length;
  const unordered_map<MSColumns, GridDistribution>& pgrid;
  bool readahead;
};

class ReadArrayTest
  : public ::testing::TestWithParam<read_array_test_params> {};

// do a read_array_test() using a communicator sized to correspond to the given
// GridDistribution; if the initial number of ranks too small, the test is
// effectively skipped (although it is reported as 'passed' by gtest)
TEST_P(ReadArrayTest, ReadPattern) {
  const struct read_array_test_params& p = GetParam();

  if (p.ms_shape[0].is_indeterminate()
      && p.traversal_order[0] != p.ms_shape[0].id())
    return;

  auto grid_size =
    accumulate(
      begin(p.ms_shape),
      end(p.ms_shape),
      1uL,
      [&pg=p.pgrid](auto&acc, auto& ax) {
        if (pg.count(ax.id()) == 0)
          return acc;
        return acc * pg.at(ax.id())(ax.length())->order();
      });
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (static_cast<unsigned long>(world_size) >= grid_size) {
    MPI_Comm testcomm;
    MPI_Comm_split(
      MPI_COMM_WORLD,
      ((static_cast<unsigned long>(rank) < grid_size) ? 0 : MPI_UNDEFINED),
      0,
      &testcomm);
    if (testcomm != MPI_COMM_NULL) {
      read_array_test(
        ".",
        testcomm,
        p.ms_order,
        p.traversal_order,
        p.buffer_length * sizeof(complex<float>),
        p.ms_shape,
        p.max_ms_length,
        p.pgrid,
        p.readahead);
      MPI_Comm_free(&testcomm);
    }
  }
}

// MS data column shape
vector<ColumnAxisBase<MSColumns> > ms_shape {
  ColumnAxis<MSColumns, MSColumns::time>(ntim),
    ColumnAxis<MSColumns, MSColumns::spectral_window>(nspw),
    ColumnAxis<MSColumns, MSColumns::baseline>(nbal),
    ColumnAxis<MSColumns, MSColumns::channel>(nch),
    ColumnAxis<MSColumns, MSColumns::polarization_product>(npol)
    };

// MS data column shape with indeterminate outer axis
vector<ColumnAxisBase<MSColumns> > ms_shape_u {
  ColumnAxis<MSColumns, MSColumns::time>(),
  ColumnAxis<MSColumns, MSColumns::spectral_window>(nspw),
  ColumnAxis<MSColumns, MSColumns::baseline>(nbal),
  ColumnAxis<MSColumns, MSColumns::channel>(nch),
  ColumnAxis<MSColumns, MSColumns::polarization_product>(npol)
};

size_t max_ms_length = ntim * nspw * nbal * nch * npol;

// read buffer lengths to test
size_t buffer_lengths[] {
  max_ms_length,
  max_ms_length / ntim,
  max_ms_length / nch,
  npol * nch
};

// process grids to test
unordered_map<MSColumns, GridDistribution> pgrid1;

unordered_map<MSColumns, GridDistribution> pgrid2 {
  {MSColumns::spectral_window, GridDistributionFactory::block_sequence(
      {{{0, 1}, {3, 1}, {4, 0}},
       {{1, 3}, {4, 0}}})}
};

unordered_map<MSColumns, GridDistribution> pgrid4 {
  {MSColumns::spectral_window, GridDistributionFactory::cyclic(1, 2) },
  {MSColumns::channel, GridDistributionFactory::cyclic(3, 2) }
};

// MS data column traversal orders to test
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

//
// helper functions for gtest output formatting
//

ostream &
operator <<(ostream& out, const read_array_test_params& p) {
  out << "(ms_order: " << p.ms_order;
  out << "; traversal_order: ";
  const char *sep = "";
  for_each(
    begin(p.traversal_order),
    end(p.traversal_order),
    [&](auto& col) {
      out << sep << mscol_nickname(col);
      sep = ",";
    });
  out << "; buffer_length: " << p.buffer_length;
  out << "; ms_shape: ";
  sep = "";
  for_each(
    begin(p.ms_shape),
    end(p.ms_shape),
    [&](auto& ax) {
      out << sep << mscol_nickname(ax.id())
          << (ax.length() ? to_string(ax.length().value()) : "-");
      sep = ",";
    });
  if (&p.pgrid == &pgrid4)
    out << "; pgrid4";
  else if (&p.pgrid == &pgrid2)
    out << "; pgrid2";
  else if (&p.pgrid == &pgrid1)
    out << "; pgrid1";
  else
    assert(false);
  out << "; readahead: " << p.readahead
      << ")";

  return out;
}

//
// test definitions
//
// Use macros to create the sequence of test definitions since there are many.
// This creates a few that we don't really test, but TEST_P(RArrayTest,
// ReadPattern) effectively skips those.

#define TP(mso, toi, bli, mss, pgn, ra)       \
  { mso,                                      \
    traversal_orders[toi],                    \
    buffer_lengths[bli],                      \
    mss,                                      \
    max_ms_length,                            \
    pgrid##pgn,                               \
    ra }

#define TP_RA(mso, toi, bli, mss, pgn)          \
  TP(mso, toi, bli, mss, pgn, false),           \
  TP(mso, toi, bli, mss, pgn, true)

#define TP_PGN(mso, toi, bli, mss)              \
  TP_RA(mso, toi, bli, mss, 4),                 \
  TP_RA(mso, toi, bli, mss, 2),                 \
  TP_RA(mso, toi, bli, mss, 1)

#define TP_MSS(mso, toi, bli)                   \
  TP_PGN(mso, toi, bli, ms_shape),              \
  TP_PGN(mso, toi, bli, ms_shape_u)

#define TP_BLI(mso, toi) \
  TP_MSS(mso, toi, 0), \
  TP_MSS(mso, toi, 1), \
  TP_MSS(mso, toi, 2), \
  TP_MSS(mso, toi, 3)

#define TP_TOI(mso) \
  TP_BLI(mso, 0), \
  TP_BLI(mso, 1), \
  TP_BLI(mso, 2), \
  TP_BLI(mso, 3)

struct read_array_test_params testdefs[] = {
  TP_TOI(true),
  TP_TOI(false)
};

INSTANTIATE_TEST_CASE_P(
  Reads,
  ReadArrayTest,
  ::testing::ValuesIn(testdefs));

int
main(int argc, char *argv[]) {

  struct EmptyTestEventListener
    : ::testing::EmptyTestEventListener {};

  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  set_throw_exception_errhandler(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank > 0) {
    // silence output in these ranks
    ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new EmptyTestEventListener);
  }
  auto result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
