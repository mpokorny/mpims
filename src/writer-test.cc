/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
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

  size_t max_buffer_size =
    ntim * nspw * nbal * nch * npol * sizeof(complex<float>);

  vector<size_t> buffer_sizes = {
    max_buffer_size,
    max_buffer_size / ntim,
    max_buffer_size / nch,
    npol * nch * sizeof(complex<float>)
  };

  // unordered_map<MSColumns, DataDistribution> pgrid;

  unordered_map<MSColumns, DataDistribution> pgrid = {
    {MSColumns::spectral_window, DataDistribution { 2, 1 } },
    {MSColumns::channel, DataDistribution { 2, 3 } }
  };

  unordered_map<MSColumns, DataDistribution> read_pgrid;

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
          if (pgrid.count(ms_top) > 0) {
            auto stride =
              pgrid[ms_top].num_processes * pgrid[ms_top].block_size;
            dims[ms_top] = ((dims[ms_top] + (stride - 1)) / stride) * stride; 
          }
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
        try {
          MPI_Bcast(
            const_cast<char*>(path.c_str()),
            path.size(),
            MPI_CHAR,
            0,
            MPI_COMM_WORLD);

          {
            Writer writer =
              Writer::begin(
                path,
                "external32",
                amode,
                MPI_COMM_WORLD,
                MPI_INFO_NULL,
                ms_shape,
                tvo,
                pgrid,
                bs);
            while (writer != Writer::end()) {
              bool done = false;
              auto indices = writer.indices();
              assert(indices.size() == dims.size() || indices.size() == 0);
              if (amode == AMode::ReadWrite
                  || indices.size() == 0
                  || indices[0].min_index() < dims[ms_top]) {
                if (writer.buffer_length() > 0) {
                  MSArray array(writer.buffer_length());
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
            {
              auto reader =
                Reader::begin(
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
              while (reader != Reader::end()) {
                auto& array = *reader;
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
            }
            cout << output.str();
            remove(path.c_str());
          }
        } catch (...) {
          if (my_rank == 0)
            remove(path.c_str());
          throw;
        }
      }
    }
  }
  MPI_Finalize();

  return EXIT_SUCCESS;
}
