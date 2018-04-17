/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <sstream>
#include <unordered_set>

#include <mpims.h>
#include <ReaderBase.h>

using namespace mpims;

void
ReaderBase::init_iterparams(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  const std::vector<MSColumns>& traversal_order,
  const std::unordered_map<MSColumns, DataDistribution>& pgrid,
  int rank,
  bool debug_log,
  std::shared_ptr<std::vector<IterParams> >& iter_params) {

  std::size_t dist_size = 1;
  std::size_t array_length = 1;
  std::for_each(
    std::begin(ms_shape),
    std::end(ms_shape),
    [&] (const ColumnAxisBase<MSColumns>& ax) {
      // find index of this column in traversal_order
      MSColumns col = ax.id();
      auto length = ax.length();
      std::size_t traversal_index = 0;
      while (traversal_order[traversal_index] != col)
        ++traversal_index;

      // create IterParams for this axis
      std::size_t num_processes;
      std::size_t block_size;
      if (pgrid.count(col) > 0) {
        auto pg = pgrid.at(col);
        num_processes = pg.num_processes;
        block_size = pg.block_size;
      } else {
        num_processes = 1;
        block_size = 1;
      }
      std::size_t grid_len = num_processes;
      std::size_t order = (rank / dist_size) % grid_len;
      std::size_t block_len = block_size;
      assert(!length || block_len <= length.value());
      std::size_t origin = order * block_len;
      std::size_t stride = grid_len * block_len;
      std::optional<std::size_t> max_blocks =
        (length
         ? std::optional<std::size_t>(ceil(length.value(), stride))
         : std::nullopt);
      std::size_t blocked_rem =
        (length
         ? (ceil(length.value(), block_len) % grid_len)
         : 0);
      std::size_t terminal_block_len;
      std::size_t max_terminal_block_len;
      if (blocked_rem == 0) {
        terminal_block_len = block_len;
        max_terminal_block_len = block_len;
      } else if (blocked_rem == 1) {
        // assert(length.has_value());
        terminal_block_len = ((order == 0) ? (length.value() % block_len) : 0);
        max_terminal_block_len = length.value() % block_len;
      } else {
        // assert(length.has_value());
        terminal_block_len =
          ((order < blocked_rem - 1)
           ? block_len
           : ((order == blocked_rem - 1) ? (length.value() % block_len) : 0));
        max_terminal_block_len = block_len;
      }
      (*iter_params)[traversal_index] =
        IterParams { col, false, false, 0, array_length,
                     origin, stride, block_len, terminal_block_len,
                     max_terminal_block_len, length, max_blocks };
      if (debug_log) {
        auto l = (*iter_params)[traversal_index].length;
        auto mb = (*iter_params)[traversal_index].max_blocks;
        std::clog << "(" << rank << ") "
                  << mscol_nickname(col)
                  << " length: "
                  << (l ? std::to_string(l.value()) : "indeterminate")
                  << ", origin: "
                  << (*iter_params)[traversal_index].origin
                  << ", stride: "
                  << (*iter_params)[traversal_index].stride
                  << ", block_len: "
                  << (*iter_params)[traversal_index].block_len
                  << ", max_blocks: "
                  << (mb ? std::to_string(mb.value()) : "indeterminate")
                  << ", terminal_block_len: "
                  << (*iter_params)[traversal_index].terminal_block_len
                  << ", max_terminal_block_len: "
                  << (*iter_params)[traversal_index].max_terminal_block_len
                  << std::endl;
      }
      dist_size *= grid_len;
    });
}

void
ReaderBase::init_traversal_partitions(
  const std::vector<ColumnAxisBase<MSColumns> >& ms_shape,
  std::size_t& buffer_size,
  std::shared_ptr<std::vector<IterParams> >& iter_params,
  std::shared_ptr<std::optional<MSColumns> >& inner_fileview_axis,
  std::size_t value_size,
  int rank,
  bool debug_log) {

  // Compute how much of the column data can fit into a buffer of the given
  // size. To do this, we move upwards from the innermost traversal axis, adding
  // the full axis when the buffer is large enough -- these axes are marked by
  // setting the fully_in_array member of the corresponding IterParams value to
  // true. When the buffer is not large enough for the full axis, we must
  // determine exactly how many values along that axis can fit into the
  // remaining space of the buffer -- these axes are marked by setting the
  // buffer_capacity member of the corresponding IterParams value to the number
  // of values (where each value on this axis must hold an array of values
  // including all prior axes) that can be accommodated into the buffer.
  std::size_t max_buffer_length = buffer_size / value_size;
  if (max_buffer_length == 0)
    throw std::runtime_error("maximum buffer size too small");
  // array_length is the total size of data in all prior axes
  std::size_t array_length = 1;
  auto start_buffer = iter_params->rbegin();
  while (start_buffer != iter_params->rend()) {
    start_buffer->array_length = array_length;
    // len is the number of values along this axis that can be fit into the
    // remaining buffer space, with a granularity of the block_len of this axis
    std::size_t len =
      std::min(
        (max_buffer_length / start_buffer->array_length)
        / start_buffer->block_len,
        start_buffer->max_blocks.value_or(1))
      * start_buffer->block_len;
    // full_len is the most values any rank will need on this axis
    std::optional<std::size_t> full_len = start_buffer->max_accessible_length();
    if (len == 0) {
      // unable to fit even a single block of values along this axis, so we go
      // back to the previous axis, then clear fully_in_array and set
      // buffer_capacity in order that the outermost axis that is at least
      // partially in a single buffer always has a strictly positive
      // buffer_capacity
      if (start_buffer != iter_params->rbegin()) {
        --start_buffer;
        // assert(start_buffer->max_blocks);
        start_buffer->buffer_capacity =
          start_buffer->max_blocks.value() * start_buffer->block_len;
        start_buffer->fully_in_array = false;
        break;
      } else {
        throw std::runtime_error("maximum buffer size too small");
      }
    } else { /* len > 0 */
      start_buffer->buffer_capacity = len;
      if (start_buffer != iter_params->rbegin()) {
        // since we were able to fit at least one value on this axis, the
        // previous axis is necessarily entirely within the buffer
        auto prev_buffer = start_buffer - 1;
        prev_buffer->fully_in_array = true;
        prev_buffer->buffer_capacity = 0;
      }
    }
    // we're done whenever the number of values in the buffer on this axis is
    // less than full_len (note that we use full_len, and not
    // start_buffer->accessible_length(), because the latter is not constant
    // across ranks, and the outcome of this method needs to be invariant across
    // ranks)
    if (!full_len || len < full_len.value())
      break;
    // increase the array size by a factor of full_len
    array_length *= full_len.value();
    ++start_buffer;
  }
  // when all axes can fit into the buffer, we need to adjust the
  // buffer_capacity and fully_in_array values of the top-level IterParams (in
  // order to maintain the property that the outermost axis that can at least
  // partially fit into the buffer has a positive buffer_capacity value, and
  // a false fully_in_array value)
  if (start_buffer == iter_params->rend()) {
    --start_buffer;
    // assert(start_buffer->max_blocks);
    start_buffer->buffer_capacity =
      start_buffer->max_blocks.value() * start_buffer->block_len;
    start_buffer->fully_in_array = false;
  }

  assert(start_buffer->buffer_capacity > 0);
  assert(start_buffer == iter_params->rbegin()
         || (start_buffer - 1)->fully_in_array);
  assert(start_buffer->buffer_capacity * start_buffer->array_length
         <= max_buffer_length);
  buffer_size =
    start_buffer->buffer_capacity
    * start_buffer->array_length
    * value_size;

  if (debug_log) {
    std::ostringstream out;
    std::for_each(
      std::begin(*iter_params),
      std::end(*iter_params),
      [&out, &rank](auto& ip) {
        out << "(" << rank << ") "
            << mscol_nickname(ip.axis)
            << " capacity " << ip.buffer_capacity
            << ", fully_in " << ip.fully_in_array
            << ", array_len " << ip.array_length
            << std::endl;
      });
    std::clog << out.str();
  }

  // determine which axes are out of order with respect to the MS order
  std::unordered_set<MSColumns> out_of_order;
  {
    auto ip = iter_params->crbegin();
    auto ms = ms_shape.crbegin();
    while (ms != ms_shape.crend()) {
      if (ms->id() == ip->axis)
        ++ip;
      else
        out_of_order.insert(ms->id());
      ++ms;
    }
  }

  // Determine the axis at which the traversal order is incompatible with the MS
  // order, with the knowledge that out of order traversal is OK if the
  // reordering can be done in memory (that, is within a single buffer). This
  // will define the axis on which the fileview needs to be set. Note that the
  // fileview may vary with index when the blocking and data distribution result
  // in a non-uniform final block (i.e, when the buffer_capacity is large enough
  // that it covers the terminal block, and terminal_block_len is different from
  // block_len on at least one rank). This potentially oddball fileview is
  // termed a "tail fileview", since it only occurs at the tail end of the axis
  // at which the fileview is set.
  {
    bool ooo = false; /* out-of-order flag */
    auto ip = iter_params->rbegin();
    while (ip != iter_params->rend()) {
      if (!*inner_fileview_axis) {
        ooo = ooo || out_of_order.count(ip->axis) > 0;
        // inner_fileview_axis is set if we've reached the out of order
        // condition and the entire axis is not held in a single buffer, or the
        // axis at which buffer_capacity is set has a non-uniform final block
        if ((ooo
             && !(ip->fully_in_array
                  || (ip->max_blocks
                      && (ip->buffer_capacity >= ip->max_accessible_length()))))
            || !std::get<1>(tail_buffer_blocks(*ip)))
          *inner_fileview_axis = ip->axis;
      }
      ip->within_fileview = !inner_fileview_axis->has_value();
      ++ip;
    }
  }
}

std::tuple<
  std::unique_ptr<MPI_Datatype, DatatypeDeleter>,
  std::size_t>
ReaderBase::vector_datatype(
  MPI_Aint value_extent,
  std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
  std::size_t dt_extent,
  std::size_t offset,
  std::size_t num_blocks,
  std::size_t block_len,
  std::size_t terminal_block_len,
  std::size_t stride,
  std::size_t len,
  int rank,
  bool debug_log) {

  std::ostringstream oss;
  if (debug_log) {
    oss << "(" << rank << ") "
        << "vector_datatype(ve " << value_extent
        << ", de " << dt_extent
        << ", of " << offset
        << ", nb " << num_blocks
        << ", bl " << block_len
        << ", tb " << terminal_block_len
        << ", st " << stride
        << ", ln " << len
        << ")";
  }
  auto result_dt = datatype();
  MPI_Aint result_dt_extent = len * dt_extent * value_extent;
  auto nb = num_blocks;
  if (terminal_block_len == 0) {
    --nb;
    terminal_block_len = block_len;
  }
  if (nb * block_len > 1 || offset > 0) {
    if (block_len == terminal_block_len && offset == 0) {
      if (block_len == stride) {
        if (debug_log)
          oss << "; contiguous " << nb * block_len;
        MPI_Type_contiguous(nb * block_len, *dt, result_dt.get());
      } else {
        if (debug_log)
          oss << "; vector " << nb
              << "," << block_len
              << "," << stride;
        MPI_Type_vector(nb, block_len, stride, *dt, result_dt.get());
      }
    } else {
      auto blocklengths = std::make_unique<int[]>(nb);
      auto displacements = std::make_unique<int[]>(nb);
      for (std::size_t i = 0; i < nb; ++i) {
        blocklengths[i] = block_len;
        displacements[i] = offset + i * stride;
      }
      blocklengths[nb - 1] = terminal_block_len;
      if (debug_log)
        oss << "; indexed " << nb
            << "," << block_len
            << "," << terminal_block_len
            << "," << offset
            << "," << stride;
      MPI_Type_indexed(
        nb,
        blocklengths.get(),
        displacements.get(),
        *dt,
        result_dt.get());
    }
    dt = std::move(result_dt);
    result_dt = datatype();
  }
  MPI_Type_create_resized(*dt, 0, result_dt_extent, result_dt.get());
  if (debug_log)
    oss << "; resize " << result_dt_extent;
  if (debug_log) {
    oss << std::endl;
    std::clog << oss.str();
  }
  return std::make_tuple(std::move(result_dt), result_dt_extent / value_extent);
}

std::tuple<std::unique_ptr<MPI_Datatype, DatatypeDeleter>, std::size_t, bool>
ReaderBase::compound_datatype(
  MPI_Aint value_extent,
  std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
  std::size_t dt_extent,
  std::size_t offset,
  std::size_t stride,
  std::size_t num_blocks,
  std::size_t block_len,
  std::size_t terminal_block_len,
  const std::optional<std::size_t>& len,
  int rank,
  bool debug_log) {

  std::unique_ptr<MPI_Datatype, DatatypeDeleter> result_dt;
  std::size_t result_dt_extent;

  // create (blocked) vector of fileview_datatype elements
  std::tie(result_dt, result_dt_extent) =
    vector_datatype(
      value_extent,
      dt,
      dt_extent,
      offset,
      num_blocks,
      block_len,
      terminal_block_len,
      stride,
      len.value_or(stride),
      rank,
      debug_log);
  return
    std::make_tuple(std::move(result_dt), result_dt_extent, !len.has_value());
}

std::tuple<std::optional<std::tuple<std::size_t, std::size_t> >, bool>
ReaderBase::tail_buffer_blocks(const IterParams& ip) {
  std::optional<std::tuple<std::size_t, std::size_t> > tb;
  std::size_t tail_num_blocks, tail_terminal_block_len;
  bool uniform;
  if (!ip.fully_in_array) {
    if (ip.length) {
      std::size_t capacity = std::max(ip.buffer_capacity, 1uL);
      auto nr = ip.stride / ip.block_len;
      auto full_buffer_capacity = capacity * nr;
      auto nb = ceil(ip.length.value(), full_buffer_capacity);
      auto tail_origin = full_buffer_capacity * (nb - 1);
      auto tail_rem = ip.length.value() - tail_origin;
      tail_num_blocks = ceil(tail_rem, ip.stride);
      auto terminal_rem = tail_rem % ip.stride;
      if (ip.origin < terminal_rem) {
        auto next = ip.origin + ip.block_len;
        if (next >= terminal_rem)
          tail_terminal_block_len = terminal_rem - ip.origin;
        else
          tail_terminal_block_len = ip.block_len;
      } else {
        tail_terminal_block_len = 0;
      }
      uniform = tail_rem % ip.stride == 0;
      tb = std::make_tuple(tail_num_blocks, tail_terminal_block_len);
    } else {
      tb = std::nullopt;
      uniform = true;
    }
  } else {
    tb = std::make_tuple(0, 0);
    uniform = true;
  }
  return std::make_tuple(tb, uniform);
}

std::unique_ptr<std::vector<IndexBlockSequenceMap<MSColumns> > >
ReaderBase::make_index_block_sequences(
  const std::shared_ptr<const std::vector<IterParams> >& iter_params) {
  std::unique_ptr<std::vector<IndexBlockSequenceMap<MSColumns> > > result(
    new std::vector<IndexBlockSequenceMap<MSColumns> >());
  std::for_each(
    std::begin(*iter_params),
    std::end(*iter_params),
    [&result](const IterParams& ip) {
      std::vector<std::vector<IndexBlock> > blocks;
      if (ip.fully_in_array || ip.buffer_capacity > 0) {
        auto accessible_length = ip.accessible_length();
        std::vector<IndexBlock> merged_blocks;
        if (accessible_length) {
          // merge contiguous blocks
          std::size_t start = ip.origin;
          std::size_t end = ip.origin + ip.block_len;
          for (std::size_t b = 1; b < ip.max_blocks.value(); ++b) {
            std::size_t s = ip.origin + b * ip.stride;
            if (s > end) {
              merged_blocks.emplace_back(start, end - start);
              start = s;
            }
            end = s + ip.block_len;
          }
          end -= ip.block_len - ip.terminal_block_len;
          merged_blocks.emplace_back(start, end - start);
        } else {
          // create enough blocks to fit buffer_capacity, merging contiguous
          // blocks
          assert(ip.buffer_capacity % ip.block_len == 0);
          auto nb = ip.buffer_capacity / ip.block_len;
          auto stride = nb * ip.stride;
          std::size_t start = ip.origin;
          std::size_t end = ip.origin + ip.block_len;
          for (std::size_t b = 1; b < nb; ++b) {
            std::size_t s = start + b * ip.stride;
            if (s > end) {
              merged_blocks.emplace_back(start, end - start, stride);
              start = s;
            }
            end = s + ip.block_len;
          }
          merged_blocks.emplace_back(start, end - start, stride);
        }

        // when entire axis doesn't fit into the array, we might have to split
        // blocks
        if (!ip.fully_in_array
            && (accessible_length
                && ip.accessible_length().value() > ip.buffer_capacity)) {
          std::size_t rem = ip.buffer_capacity;
          std::vector<IndexBlock> ibs;
          std::for_each(
            std::begin(merged_blocks),
            std::end(merged_blocks),
            [&ip, &blocks, &rem, &ibs](IndexBlock& blk) {
              while (blk.m_length > 0) {
                std::size_t len = std::min(rem, blk.m_length);
                ibs.emplace_back(blk.m_index, len);
                blk = IndexBlock(blk.m_index + len, blk.m_length - len);
                rem -= len;
                if (rem == 0) {
                  blocks.push_back(ibs);
                  ibs.clear();
                  rem = ip.buffer_capacity;
                }
              }
            });
          if (ibs.size() > 0)
            blocks.push_back(ibs);
        } else {
          blocks.push_back(merged_blocks);
        }
      } else {
        blocks.push_back(std::vector<IndexBlock>{ IndexBlock{ ip.origin, 1 } });
      }
      result->emplace_back(ip.axis, blocks);
    });
  return result;
}

const IterParams*
ReaderBase::find_iter_params(
  const std::shared_ptr<const std::vector<IterParams> >& iter_params,
  MSColumns col) {

  auto ip =
    std::find_if(
      std::begin(*iter_params),
      std::end(*iter_params),
      [&col](auto& ip) {
        return ip.axis == col;
      });
  if (ip != std::end(*iter_params))
    return &*ip;
  return nullptr;
}
