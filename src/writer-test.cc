#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

#include <mpi.h>
#include <unistd.h>

#include <mpims.h>
#include <ColumnAxis.h>
#include <Reader.h>
#include <Writer.h>

using namespace mpims;

using namespace std;

constexpr auto ntim = 4;
constexpr auto nbits_tim = 2;
constexpr auto nspw = 4;
constexpr auto nbits_spw = 2;
constexpr auto nbal = 3 /*6*/;
constexpr auto nbits_bal = 3;
constexpr auto nch = 13;
// nbits_ch is larger than ceil(log2(13)) since one of the test process grids
// distributes the elements such that the effective nch is 18
constexpr auto nbits_ch = 5;
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
  vector<complex<float> >& full_array,
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
  vector<complex<float> >& full_array,
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
  vector<complex<float> >& full_array,
  shared_ptr<ArrayIndexer<MSColumns> >& full_array_indexer,
  bool& index_out_of_range,
  ostringstream& output) {

  if (!array.buffer())
    return false;
  auto indexes = array.blocks();

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
  return result;
}

void
writeit(
  complex<float>** buffer,
  stack<IndexBlockSequence<MSColumns> >& index_blocks,
  unordered_map<MSColumns, size_t>& indices) {

  MSColumns col = index_blocks.top().m_axis;
  if (index_blocks.size() == 1) {
    complex<float>* b = *buffer;
    for (auto& ib : index_blocks.top().m_blocks) {
      for (size_t i = 0; i < ib.m_length; ++i) {
        indices[col] = ib.m_index + i;
        encode_vis(
          *b++,
          indices[MSColumns::time],
          indices[MSColumns::spectral_window],
          indices[MSColumns::baseline],
          indices[MSColumns::channel],
          indices[MSColumns::polarization_product]);
      }
    }
    *buffer = b;
  } else {
    IndexBlockSequence<MSColumns> top = index_blocks.top();
    index_blocks.pop();
    for (auto& ib : top.m_blocks) {
      for (size_t i = 0; i < ib.m_length; ++i) {
        indices[col] = ib.m_index + i;
        writeit(buffer, index_blocks, indices);
      }
    }
    index_blocks.push(top);
  }
}

void
write_buffer(
  optional<complex<float>*> buffer,
  const vector<IndexBlockSequence<MSColumns> >& ibs) {

  if (!buffer)
    return;

  auto buff = buffer.value();

  stack<IndexBlockSequence<MSColumns> > index_blocks;
  std::for_each(
    ibs.crbegin(),
    ibs.crend(),
    [&index_blocks](auto& ib) { index_blocks.push(ib); });

  unordered_map<MSColumns, size_t> indices;
  writeit(&buff, index_blocks, indices);
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
    cerr << "usage: " << argv[0] << " <directory>" << endl;
    return -1;
  }

  MPI_Init(&argc, &argv);
  set_throw_exception_errhandler(MPI_COMM_WORLD);

  unordered_map<MSColumns, size_t> dimensions {
    {MSColumns::time, ntim},
    {MSColumns::spectral_window, nspw},
    {MSColumns::baseline, nbal},
    {MSColumns::channel, nch},
    {MSColumns::polarization_product, npol}
  };

  vector<vector<MSColumns> > ms_axis_orders {
    {MSColumns::time, MSColumns::spectral_window, MSColumns::baseline,
        MSColumns::channel, MSColumns::polarization_product},
    {MSColumns::spectral_window, MSColumns::time, MSColumns::baseline,
        MSColumns::channel, MSColumns::polarization_product},
    {MSColumns::channel, MSColumns::spectral_window,
        MSColumns::time, MSColumns::baseline, MSColumns::polarization_product},
    {MSColumns::polarization_product, MSColumns::spectral_window,
        MSColumns::time, MSColumns::baseline, MSColumns::channel}
  };

  vector<MSColumns> read_order = ms_axis_orders[0];

  size_t max_buffer_length = ntim * nspw * nbal * nch * npol;

  size_t max_buffer_size = max_buffer_length * sizeof(complex<float>);

  vector<size_t> buffer_sizes = {
    max_buffer_size,
    max_buffer_size / ntim,
    max_buffer_size / nch,
    npol * nch * sizeof(complex<float>)
  };

  unordered_map<MSColumns, GridDistribution> pgrid1;

  unordered_map<MSColumns, GridDistribution> pgrid2 {
    {MSColumns::spectral_window, GridDistributionFactory::block_sequence(
        {{{0, 1}, {3, 1}, {4, 0}},
         {{1, 2}, {4, 0}}})}
  };

  unordered_map<MSColumns, GridDistribution> pgrid4 = {
    {MSColumns::spectral_window, GridDistributionFactory::cyclic(1, 2) },
    {MSColumns::channel, GridDistributionFactory::cyclic(3, 2) }
  };

  unordered_map<int, unordered_map<MSColumns, GridDistribution> > pgrids {
    {1, pgrid1},
    {2, pgrid2},
    {4, pgrid4}
  };

  unordered_map<MSColumns, GridDistribution> read_pgrid;

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int grid_size = world_size;
  auto pg = pgrids.find(grid_size);
  while (pg == end(pgrids))
    pg = pgrids.find(--grid_size);
  auto pgrid = pg->second;

  for (auto& msao : ms_axis_orders) {
    for (auto& bs : buffer_sizes) {
      for (auto& tvo : ms_axis_orders) {

        vector<ColumnAxisBase<MSColumns> > ms_shape;
        transform(
          begin(msao),
          end(msao),
          back_inserter(ms_shape),
          [&dimensions](auto& col) {
            return ColumnAxisBase<MSColumns>(
              static_cast<unsigned>(col),
              dimensions[col]);
          });

        unordered_map<MSColumns,std::size_t> dims(dimensions);
        MSColumns ms_top = ms_shape[0].id();
        AMode amode;
        if (ms_top == tvo[0]) {
          amode = AMode::WriteOnly;
          ms_shape[0] =
            ColumnAxisBase<MSColumns>(static_cast<unsigned>(ms_top));
        } else {
          amode = AMode::ReadWrite;
        }
        MPI_Aint ms_top_len = dims[ms_top];
        ostringstream output;
        output << "========= ms order "
               << colnames(msao)
               << "; traversal order "
               << colnames(tvo)
               << "; write buffer_size "
               << num_elements(bs)
               << " ========="
               << endl;
        string path(argv[1]);
        path += "/wtXXXXXX";
        if (my_rank == 0) {
          int fd = mkstemp(const_cast<char*>(path.c_str()));
          if (fd == -1) {
            cerr << "Failed to open temporary file: "
                 << strerror(errno)
                 << endl;
            return EXIT_FAILURE;
          }
          close(fd);
        }

        MPI_Bcast(
          const_cast<char*>(path.c_str()),
          path.size(),
          MPI_CHAR,
          0,
          MPI_COMM_WORLD);

        {
          CxFltWriter writer =
            CxFltWriter::begin(
              path,
              "external32",
              amode,
              MPI_COMM_WORLD,
              MPI_INFO_NULL,
              ms_shape,
              tvo,
              pgrid,
              bs);
          while (writer != CxFltWriter::end()) {
            bool done = false;
            auto indices = writer.indices();
            assert(indices.size() == dims.size() || indices.size() == 0);
            bool inc =
              amode == AMode::ReadWrite
              || indices.size() == 0
              || indices[0].min_index() < dims[ms_top];
            MPI_Allreduce(
              MPI_IN_PLACE,
              &inc,
              1,
              MPI_CXX_BOOL,
              MPI_LOR,
              MPI_COMM_WORLD);
            if (inc) {
              if (writer.buffer_length() > 0) {
                CxFltMSArray array(writer.buffer_length());
                write_buffer(array.buffer(), indices);
                *writer = move(array);
              }
              ++writer;
            } else {
              done = true;
            }
            if (done)
              writer.interrupt();
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // read back
        if (my_rank == 0) {
          if (amode == AMode::WriteOnly)
            ms_shape[0] =
              ColumnAxisBase<MSColumns>(
                static_cast<unsigned>(ms_top),
                ms_top_len);

          auto full_array_idx =
            ArrayIndexer<MSColumns>::of(ArrayOrder::row_major, ms_shape);
          vector<complex<float> > full_array(
            accumulate(
              begin(ms_shape),
              end(ms_shape),
              1,
              [](auto& acc, auto& ax) {
                return acc * ax.length().value();
              }));
          fill(begin(full_array), end(full_array), NAN);
          bool index_oor = false;

          {
            auto reader =
              CxFltReader::begin(
                path,
                "external32",
                MPI_COMM_SELF,
                MPI_INFO_NULL,
                ms_shape,
                read_order,
                true,
                read_pgrid,
                max_buffer_size,
                false);

            while (reader != CxFltReader::end()) {
              auto& array = *reader;
              if (array.buffer()) {
                if (cb(
                      array,
                      full_array,
                      full_array_idx,
                      index_oor,
                      output))
                  ++reader;
                else
                  reader.interrupt();
              }
              else {
                ++reader;
              }
            }
          }
          cout << output.str();
          if (index_oor)
            cout << "---- error: out of range array indexes ---- "
                 << endl;
          size_t num_missing =
            count_if(
              begin(full_array),
              end(full_array),
              [](auto& c) { return isnan(c); });
          if (num_missing > 0)
            cout << "++++ error: "
                 << num_missing
                 << " missing elements ++++" << endl;
          remove(path.c_str());
        }
      }
    }
  }
  MPI_Finalize();

  return EXIT_SUCCESS;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
