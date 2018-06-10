#ifndef TRAVERSAL_STATE_H_
#define TRAVERSAL_STATE_H_

#include <deque>
#include <sstream>
#include <memory>
#include <vector>

#include <mpi.h>

#include <ArrayIndexer.h>
#include <AxisIter.h>
#include <IndexBlock.h>
#include <IterParams.h>
#include <MSColumns.h>
#include <ReaderBase.h>

namespace mpims {

// TraversalState maintains the state used to traverse the MS in the order
// provided by the client. If one considers MS traversal as stack of
// traversals along every axis with recursion to get to the next axis, this
// structure contains the information needed for traversal after flattening
// such a recursive iteration into a single loop.
struct TraversalState {

  TraversalState()
    : cont(false)
    , global_eof(true) {
  }

  TraversalState(
    const std::shared_ptr<const std::vector<IterParams> >& iter_params_,
    MSColumns outer_ms_axis,
    std::size_t outer_ms_length,
    bool in_tail_,
    MSColumns outer_full_array_axis_,
    const std::shared_ptr<const MPI_Datatype>& full_buffer_datatype,
    unsigned full_buffer_dt_count,
    const std::shared_ptr<const MPI_Datatype>& tail_buffer_datatype,
    unsigned tail_buffer_dt_count,
    const std::shared_ptr<const MPI_Datatype>& full_fileview_datatype,
    const std::shared_ptr<const MPI_Datatype>& tail_fileview_datatype)
  : iter_params(iter_params_)
  , block_maps(ReaderBase::make_index_block_sequences(iter_params_))
  , count(0)
  , max_count(0)
  , in_tail(in_tail_)
  , outer_full_array_axis(outer_full_array_axis_)
  , m_full_buffer_datatype(full_buffer_datatype)
  , m_full_buffer_dt_count(full_buffer_dt_count)
  , m_tail_buffer_datatype(tail_buffer_datatype)
  , m_tail_buffer_dt_count(tail_buffer_dt_count)
  , m_full_fileview_datatype(full_fileview_datatype)
  , m_tail_fileview_datatype(tail_fileview_datatype) {

    global_eof = false;
    const IterParams* init_params = &(*iter_params)[0];
    axis_iters.emplace_back(
      std::shared_ptr<const IterParams>(iter_params, init_params),
      !init_params->max_blocks || init_params->max_blocks > 0);
    std::for_each(
      std::begin(*iter_params),
      std::end(*iter_params),
      [this](const IterParams& ip) {
        data_index[ip.axis] = ip.origin;
      });

    if (axis_iters.front().params->axis == outer_ms_axis) {
      eof_axis_iters.push_back(axis_iters.front());
      auto& eai = eof_axis_iters.front();
      max_count = std::max(eai.params->buffer_capacity, 1uL);
      while (!eai.at_end && eai.index < outer_ms_length)
        eai.increment();
    }
    cont = true;
  }

  //EOF condition flag
  bool
  eof() const {
    if (axis_iters.empty() || global_eof)
      return true;
    if (eof_axis_iters.empty())
      return false;
    auto ai = std::rbegin(axis_iters);
    auto ai_end = std::rend(axis_iters);
    auto eai = std::rbegin(eof_axis_iters);
    auto eai_end = std::rend(eof_axis_iters);
    bool result = false;
    auto num_undefined = axis_iters.size() - eof_axis_iters.size();
    while (num_undefined-- > 0 && ai != ai_end)
      ++ai;
    while (!result && ai != ai_end && eai != eai_end) {
      result = ai->index >= eai->index;
      ++ai;
      ++eai;
    }
    return result;
  };

  bool
  at_end() const {
    return !cont || global_eof;
  }

  // continuation condition flag (to allow breaking out from iteration)
  bool cont;

  bool global_eof;

  std::shared_ptr<const std::vector<IterParams> > iter_params;

  std::shared_ptr<
    const std::vector<IndexBlockSequenceMap<MSColumns> > > block_maps;

  ArrayIndexer<MSColumns>::index data_index;

  // stack of AxisIters to maintain axis iteration indexes
  std::deque<AxisIter> axis_iters;

  std::deque<AxisIter> eof_axis_iters;

  int count;

  int max_count;

  // Is traversal currently in a tail fileview? A tail fileview differs
  // somewhat from a non-tail fileview when the iteration at the outermost
  // fileview axis doesn't fit into the same pattern at the end of the axis as
  // all the previous fileviews. This can happen when the distribution of data
  // on an axis to processes is uneven at the end of an axis.
  bool in_tail;

  // the outermost axis at which data can fully fit into a buffer
  MSColumns outer_full_array_axis;

  std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned>
  buffer_datatype() const {
    if (in_tail)
      return std::make_tuple(m_tail_buffer_datatype, m_tail_buffer_dt_count);
    else
      return std::make_tuple(m_full_buffer_datatype, m_full_buffer_dt_count);
  }

  std::shared_ptr<const MPI_Datatype>
  fileview_datatype() const {
    return (in_tail ? m_tail_fileview_datatype : m_full_fileview_datatype);
  }

  bool
  operator==(const TraversalState& other) const {
    return (
      cont == other.cont
      && count == other.count
      && (block_maps == other.block_maps || *block_maps == *other.block_maps)
      && data_index == other.data_index
      && axis_iters == other.axis_iters
      && eof_axis_iters == other.eof_axis_iters
      && in_tail == other.in_tail
      && m_tail_buffer_dt_count == other.m_tail_buffer_dt_count
      && m_full_buffer_dt_count == other.m_full_buffer_dt_count
      // comparing array datatypes and fileview datatypes would be ideal, but
      // that could be a relatively expensive task, as it requires digging
      // into the contents of the datatypes; instead, since those datatypes
      // are functions of the MS shape, iteration order and outer array axis
      // alone, given the previous comparisons, we can just compare the outer
      // arrays axis values
      && outer_full_array_axis == other.outer_full_array_axis);
  }

  bool
  operator!=(const TraversalState& other) const {
    return !operator==(other);
  }

  std::vector<IndexBlockSequence<MSColumns> >
  blocks(int cnt) const {
    std::vector<IndexBlockSequence<MSColumns> > result;
    if (cnt > 0) {
      for (std::size_t i = 0; i < iter_params->size(); ++i) {
        auto& ip = (*iter_params)[i];
        if (!ip.fully_in_array && ip.buffer_capacity == 0) {
          result.emplace_back(
            ip.axis,
            std::vector{ IndexBlock(data_index.at(ip.axis), 1) });
        } else {
          auto& map = (*block_maps)[i];
          auto ibs = map[data_index.at(ip.axis)];
          ibs.trim();
          result.push_back(ibs);
        }
      }
    }
    return result;
  }

  void
  advance_to_buffer_end() {
    if (!axis_iters.empty()) {
      AxisIter* axis_iter = &axis_iters.back();
      axis_iter->increment(max_count);
      data_index[axis_iter->params->axis] = axis_iter->index;
      while (!axis_iters.empty() && axis_iter->at_end) {
        data_index[axis_iter->params->axis] = axis_iter->params->origin;
        axis_iters.pop_back();
        if (!axis_iters.empty()) {
          axis_iter = &axis_iters.back();
          axis_iter->increment();
          data_index[axis_iter->params->axis] = axis_iter->index;
        }
      }}
  }

  template <typename F>
  void
  advance_to_next_buffer(
    std::shared_ptr<const std::optional<MSColumns> > inner_fileview_axis,
    F at_fileview_axis) {

    count = 0;
    max_count = 0;
    while (!axis_iters.empty()) {
      AxisIter& axis_iter = axis_iters.back();
      MSColumns axis = axis_iter.params->axis;
      if (!axis_iter.at_end) {
        auto depth = axis_iters.size();
        data_index[axis] = axis_iter.index;
        if (axis_iter.params->buffer_capacity > 0) {
          max_count = static_cast<int>(axis_iter.params->buffer_capacity);
          auto nr = axis_iter.num_remaining();
          if (nr && (nr.value() < static_cast<std::size_t>(max_count))) {
            count = nr.value();
            in_tail = true;
          } else {
            count = max_count;
            in_tail = false;
          }
        } else {
          in_tail = false;
        }
        if (*inner_fileview_axis && axis == inner_fileview_axis->value())
          at_fileview_axis(axis);
        if (axis_iter.params->buffer_capacity > 0)
          return;
        const IterParams* next_params = &(*iter_params)[depth];
        axis_iters.emplace_back(
          std::shared_ptr<const IterParams>(iter_params, next_params),
          axis_iter.at_data);
      } else {
        data_index[axis] = axis_iter.params->origin;
        axis_iters.pop_back();
        if (!axis_iters.empty())
          axis_iters.back().increment();
      }
    }
  }

private:

  std::shared_ptr<const MPI_Datatype> m_full_buffer_datatype;

  unsigned m_full_buffer_dt_count;

  std::shared_ptr<const MPI_Datatype> m_tail_buffer_datatype;

  unsigned m_tail_buffer_dt_count;

  std::shared_ptr<const MPI_Datatype> m_full_fileview_datatype;

  std::shared_ptr<const MPI_Datatype> m_tail_fileview_datatype;

  std::ostringstream
  show_iters(const std::deque<AxisIter>& iters) const {
    std::ostringstream result;
    std::for_each(
      std::begin(iters),
      std::end(iters),
      [&result](const auto& ai) {
        result << mscol_nickname(ai.params->axis) << ":" << ai.index << ";";
      });
    return result;
  }
};

} // end namespace mpims

#endif // TRAVERSAL_STATE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
