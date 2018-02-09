/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
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
  const shared_ptr<complex<float> >& buffer,
  size_t& n,
  vector<pair<MSColumns, size_t> >& coords,
  vector<IndexBlockSequence<MSColumns> >::const_iterator begin,
  vector<IndexBlockSequence<MSColumns> >::const_iterator end,
  ostringstream& output)
  __attribute__((unused));

bool
checkit(
  const shared_ptr<const complex<float> >& buffer,
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
cb(
  const vector<IndexBlockSequence<MSColumns> >& indexes,
  const shared_ptr<const complex<float> >& buffer,
  ostringstream& output) {

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
  bool result = checkit(buffer, n, coords, begin(indexes), end(indexes), output);
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
  complex<float>* buffer,
  const vector<IndexBlockSequence<MSColumns> >& ibs) {

  stack<IndexBlockSequence<MSColumns> > index_blocks;
  std::for_each(
    ibs.crbegin(),
    ibs.crend(),
    [&index_blocks](auto& ib) { index_blocks.push(ib); });

  unordered_map<MSColumns, size_t> indices;
  writeit(&buffer, index_blocks, indices);
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

  ::MPI_Init(&argc, &argv);
  ::MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

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
  mpi_call(::MPI_Comm_rank, MPI_COMM_WORLD, &my_rank);
  int world_size;
  mpi_call(::MPI_Comm_size, MPI_COMM_WORLD, &world_size);

  for (auto& msao : ms_axis_orders) {
    for (auto& bs : buffer_sizes) {
      for (auto& tvo : ms_axis_orders) {
        ostringstream output;
        output << "========= ms axis order "
               << colnames(msao)
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
        mpi_call(
          ::MPI_Bcast,
          const_cast<char*>(path.c_str()),
          path.size(),
          MPI_CHAR,
          0,
          MPI_COMM_WORLD);

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

        {
          auto writer =
            Writer::begin(
              path,
              MPI_COMM_WORLD,
              MPI_INFO_NULL,
              ms_shape,
              tvo,
              pgrid,
              bs);
          while (writer != Writer::end()) {
            if (writer.buffer_length() > 0) {
              unique_ptr<complex<float> > buffer = writer.allocate_buffer();
              write_buffer(buffer.get(), writer.indices());
              *writer = move(buffer);
            }
            ++writer;
          }
        }

        // read back
        if (my_rank == 0) {
          auto reader =
            Reader::begin(
              path,
              MPI_COMM_SELF,
              MPI_INFO_NULL,
              ms_shape,
              read_order,
              true,
              read_pgrid,
              max_buffer_size,
              false);
          while (reader != Reader::end()) {
            auto array = *reader;
            if (array) {
              if (cb(reader.indices(), array, output))
                ++reader;
              else
                reader.interrupt();
            }
            else {
              ++reader;
            }
          }
          cout << output.str();
          remove(path.c_str());
        }
      }
    }
  }
  ::MPI_Finalize();

  return EXIT_SUCCESS;
}
