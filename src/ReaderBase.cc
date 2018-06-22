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
  const std::unordered_map<
  MSColumns,
  std::shared_ptr<const DataDistribution> >& pgrid,
  int rank,
  std::shared_ptr<std::vector<IterParams> >& iter_params) {

  std::size_t dist_size = 1;
  std::for_each(
    std::begin(ms_shape),
    std::end(ms_shape),
    [&] (const ColumnAxisBase<MSColumns>& ax) {
      // find index of this column in traversal_order
      MSColumns col = ax.id();
      std::size_t traversal_index = 0;
      while (traversal_order[traversal_index] != col)
        ++traversal_index;

      // create IterParams for this axis
      auto& dd = pgrid.at(col);
      std::size_t grid_len = dd->order();
      (*iter_params)[traversal_index] =
        IterParams(col, ax.length(), dd, (rank / dist_size) % grid_len);
      dist_size *= grid_len;
    });
}

void
ReaderBase::init_traversal_partitions(
  MPI_Comm comm,
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
    // remaining buffer space; subject to the number of values being a multiple
    // of the number of elements in one period (or 1 if the axis is not
    // periodic)
    auto blk_factor =  start_buffer->period_max_factor_size();
    std::size_t len =
      ((max_buffer_length / start_buffer->array_length) / blk_factor)
      * blk_factor;
    if (len == 0) {
      // unable to fit even the minimum number of values along this axis, so we
      // go back to the previous axis, then clear fully_in_array and set
      // buffer_capacity in order that the outermost axis that is at least
      // partially in a single buffer always has a strictly positive
      // buffer_capacity
      if (start_buffer != iter_params->rbegin()) {
        --start_buffer;
        start_buffer->buffer_capacity = start_buffer->size().value();
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
    // max_len is the most values any rank will need on this axis
    std::optional<std::size_t> max_len = start_buffer->max_size();
    // we're done whenever the number of values in the buffer on this axis is
    // less than max_len (note that we use max_len, and not len, because the
    // latter is not constant across ranks, and the outcome of this method needs
    // to be invariant across ranks)
    if (!max_len || len < max_len.value())
      break;
    // increase the array size by a factor of max_len
    array_length *= max_len.value();
    ++start_buffer;
  }
  // when all axes can fit into the buffer, we need to adjust the
  // buffer_capacity and fully_in_array values of the top-level IterParams (in
  // order to maintain the property that the outermost axis that can at least
  // partially fit into the buffer has a positive buffer_capacity value, and
  // a false fully_in_array value)
  if (start_buffer == iter_params->rend()) {
    --start_buffer;
    start_buffer->buffer_capacity = start_buffer->size().value();
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
  // will define the axis on which the fileview needs to be set.
  {
    bool ooo = false; // out-of-order flag
    auto ip = iter_params->rbegin();
    while (ip != iter_params->rend()) {
      if (!*inner_fileview_axis) {
        ooo = ooo || out_of_order.count(ip->axis) > 0;
        bool complete_coverage =
          ip->fully_in_array ||
          map(
            ip->max_size(),
            [ip](const auto& sz) { return sz <= ip->buffer_capacity; }).
          value_or(false);
        if (!complete_coverage
            && (ooo || !ip->selection_repeats_uniformly(comm)))
          *inner_fileview_axis = ip->axis;
      }
      ip->within_fileview = !inner_fileview_axis->has_value();
      ++ip;
    }
  }
}

std::tuple<std::unique_ptr<MPI_Datatype, DatatypeDeleter>, std::size_t>
ReaderBase::finite_compound_datatype(
  std::unique_ptr<MPI_Datatype, DatatypeDeleter>& dt,
  std::size_t dt_extent,
  const std::vector<finite_block_t>& blocks,
  std::size_t len,
  int rank,
  bool debug_log) {

  std::ostringstream oss;
  if (debug_log) {
    oss << "(" << rank << ") "
        << "finite_compund_datatype(de " << dt_extent
        << ", b [";
    const char* sep = "";
    std::for_each(
      std::begin(blocks),
      std::end(blocks),
      [&oss, &sep](auto& blk) {
        oss << sep << "(" << std::get<0>(blk)
            << "," << std::get<1>(blk)
            << ")";
        sep = ",";
      });
    oss << "], ln " << len
        << ")";
  }

  // scan blocks to determine whether block size and/or block stride are
  // constant
  std::optional<std::size_t> stride;
  std::optional<std::size_t> block_len;
  std::size_t num_blocks = blocks.size();
  std::size_t offset;
  std::tie(offset, block_len) = blocks[0];
  if (num_blocks == 1) {
    stride = block_len.value();
  } else {
    std::size_t prev_b0 = offset;
    std::size_t cur_b0, cur_blen;
    std::tie(cur_b0, cur_blen) = blocks[1];
    stride = cur_b0 - prev_b0;
    if (cur_blen != block_len.value())
      block_len.reset();
    std::for_each(
      std::begin(blocks) + 2,
      std::end(blocks),
      [&](auto& blk) {
        prev_b0 = cur_b0;
        std::tie(cur_b0, cur_blen) = blk;
        auto s = cur_b0 - prev_b0;
        if (stride && s != stride.value())
          stride.reset();
        if (block_len && cur_blen != block_len.value())
          block_len.reset();
      });
  }

  // create compound datatype
  auto compound_dt = datatype();
  if (block_len) {
    if (stride) {
      // these following datatypes are created without any leading offset
      if (stride == block_len) {
        //contiguous datatype
        if (debug_log)
          oss << "; contiguous " << num_blocks * block_len.value();
        MPI_Type_contiguous(
          num_blocks * block_len.value(),
          *dt,
          compound_dt.get());
      } else {
        // vector datatype
        if (debug_log)
          oss << "; vector " << num_blocks
              << "," << block_len.value()
              << "," << stride.value();
        MPI_Type_vector(
          num_blocks,
          block_len.value(),
          stride.value(),
          *dt,
          compound_dt.get());
      }
    } else  {
      // indexed block datatype
      if (debug_log) {
        oss << "; idxblk " << num_blocks
            << ", " << block_len.value() << ", [";
        const char* sep = "";
        std::for_each(
          std::begin(blocks),
          std::end(blocks),
          [&oss, &sep, &offset](auto& blk) {
            oss << sep << std::get<0>(blk);
            sep = ",";
          });
        oss << "]";
      }
      auto displacements = std::make_unique<int[]>(num_blocks);
      for (std::size_t i = 0; i < num_blocks; ++i)
        displacements[i] = std::get<0>(blocks[i]);
      MPI_Type_create_indexed_block(
        num_blocks,
        block_len.value(),
        displacements.get(),
        *dt,
        compound_dt.get());
      // offset has been accounted for, set it to zero
      offset = 0;
    }
  } else {
    // indexed datatype
    if (debug_log) {
      oss << "; indexed " << num_blocks
          << ", [";
      const char* sep = "";
      std::for_each(
        std::begin(blocks),
        std::end(blocks),
        [&oss, &sep, &offset](auto& blk) {
          oss << sep << "(" << std::get<0>(blk)
              << "," << std::get<1>(blk) << ")";
          sep = ",";
        });
      oss << "]";
    }
    auto blocklengths = std::make_unique<int[]>(num_blocks);
    auto displacements = std::make_unique<int[]>(num_blocks);
    for (std::size_t i = 0; i < num_blocks; ++i) 
      std::tie(displacements[i], blocklengths[i]) = blocks[i];
    MPI_Type_indexed(
      num_blocks,
      blocklengths.get(),
      displacements.get(),
      *dt,
      compound_dt.get());
    // offset has been accounted for, set it to zero
    offset = 0;
  }
  if (debug_log) {
    oss << "; resize "
        << offset << ", " << len
        << std::endl;
    std::clog << oss.str();
  }
  if (offset > 0) {
    auto cdt = std::move(compound_dt);
    compound_dt = datatype();
    int blocklength = 1;
    MPI_Aint displacement = offset * dt_extent;
    MPI_Type_create_hindexed(
      1,
      &blocklength,
      &displacement,
      *cdt,
      compound_dt.get());
  }
  std::size_t result_dt_extent = len * dt_extent;
  auto result_dt = datatype();
  MPI_Type_create_resized(
    *compound_dt,
    0,
    result_dt_extent,
    result_dt.get());

  return
    std::make_tuple(std::move(result_dt), result_dt_extent);
}

IterParams*
ReaderBase::find_iter_params(
  const std::shared_ptr<std::vector<IterParams> >& iter_params,
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

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
