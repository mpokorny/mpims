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
    const std::shared_ptr<const std::vector<IterParams> >& iter_params,
    MSColumns outer_ms_axis,
    std::size_t outer_ms_length,
    MSColumns outer_full_array_axis_,
    const std::function<
      const std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned>&(
        const std::size_t&)>& buffer_datatypes,
    const std::function<
      const std::shared_ptr<const MPI_Datatype>&(
        const std::vector<finite_block_t>&) >& fileview_datatypes)
  : m_iter_params(iter_params)
  , outer_full_array_axis(outer_full_array_axis_)
  , m_buffer_datatypes(buffer_datatypes)
  , m_fileview_datatypes(fileview_datatypes) {

    global_eof = false;
    axis_iters.emplace_back(this->iter_params(0), true);
    if (axis_iters.front().axis() == outer_ms_axis)
      initial_outer_length = outer_ms_length;
    cont = true;
  }

  //EOF condition
  bool
  eof() const {
    if (axis_iters.empty() || global_eof)
      return true;
    return
      map(
        initial_outer_length,
        [&](auto iol) {
          auto next_blocks = axis_iters.front().next_blocks();
          return (next_blocks.size() == 0
                  || std::get<0>(next_blocks[0]) >= iol); 
        }).value_or(false);
  };

  bool
  at_end() const {
    return !cont || global_eof;
  }

  // continuation condition flag (to allow breaking out from iteration)
  bool cont;

  bool global_eof;

  std::shared_ptr<const std::vector<IterParams> > m_iter_params;

  std::unordered_map<MSColumns, std::vector<finite_block_t> > data_blocks;

  bool
  at_data() const {
    return
      std::all_of(
        std::begin(data_blocks),
        std::end(data_blocks),
        [](auto& cblk) {
          return std::get<1>(cblk).size() > 0;
        });
  }

  // stack of AxisIters to maintain axis iteration indexes
  std::deque<AxisIter> axis_iters;

  // the outermost axis at which data can fully fit into a buffer
  MSColumns outer_full_array_axis;

  std::optional<std::size_t> initial_outer_length;

  std::shared_ptr<const IterParams>
  iter_params(std::size_t i) {
    return
      std::shared_ptr<const IterParams>(m_iter_params, &(*m_iter_params)[i]);
  }

  const std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned>&
  buffer_datatype() const {
    static std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned>
      zilch(datatype(), 0);

    // find data_blocks at buffer axis
    const std::vector<finite_block_t>* blks = nullptr;
    for (std::size_t i = 0; i < axis_iters.size() && !blks; ++i) {
      if (axis_iters[i].at_buffer())
        blks = &data_blocks.at(axis_iters[i].axis());
    }
    if (blks) {
      // get buffer datatype, based on number of elements in blocks
      std::size_t count = 0;
      std::for_each(
        std::begin(*blks),
        std::end(*blks),
        [&count](auto& blk) {
          count += std::get<1>(blk);
        });
      return m_buffer_datatypes(count);
    } else {
      return zilch;
    }
  }

  const std::shared_ptr<const MPI_Datatype>&
  fileview_datatype(
    const std::shared_ptr<const std::optional<MSColumns> >& fileview_axis)
    const {
    // find data_blocks at fileview axis
    std::vector<finite_block_t> blks;
    if (*fileview_axis) {
      const std::vector<finite_block_t>* blks_p = nullptr;
      for (std::size_t i = 0; i < axis_iters.size() && !blks_p; ++i) {
        if (axis_iters[i].axis() == fileview_axis->value())
          blks_p = &data_blocks.at(axis_iters[i].axis());
      }
      assert(blks_p);

      std::size_t origin = std::get<0>((*blks_p)[0]);
      std::transform(
        std::begin(*blks_p),
        std::end(*blks_p),
        std::back_inserter(blks),
        [&origin](auto& blk) {
          std::size_t b0, blen;
          std::tie(b0, blen) = blk;
          assert(b0 >= origin);
          return std::make_tuple(b0 - origin, blen);
        });
    } 
    return m_fileview_datatypes(blks); 
  }

  bool
  operator==(const TraversalState& other) const {
    return (
      cont == other.cont
      && data_blocks == other.data_blocks
      && axis_iters == other.axis_iters
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
  blocks() const {
    std::vector<IndexBlockSequence<MSColumns> > result;
    if (at_data()) {
      for (std::size_t i = 0; i < m_iter_params->size(); ++i) {
        auto& ip = (*m_iter_params)[i];
        std::vector<IndexBlock> ibs;
        std::transform(
          std::begin(data_blocks.at(ip.axis)),
          std::end(data_blocks.at(ip.axis)),
          std::back_inserter(ibs),
          [&](auto& blk) {
            std::size_t b0, blen;
            std::tie(b0, blen) = blk;
            return IndexBlock(b0, blen);
          });
        result.push_back(IndexBlockSequence(ip.axis, ibs));
      }
    }
    return result;
  }

  void
  advance_to_buffer_end() {
    while (!axis_iters.empty() && axis_iters.back().at_end()) 
      axis_iters.pop_back(); 
  }

  template <typename F>
  void
  advance_to_next_buffer(
    const std::shared_ptr<const std::optional<MSColumns> >& inner_fileview_axis,
    F at_fileview_axis) {

    while (!axis_iters.empty()) {
      AxisIter& axis_iter = axis_iters.back();
      MSColumns axis = axis_iter.axis();
      if (!axis_iter.at_end()) {
        bool at_data = axis_iter.at_data();
        data_blocks[axis] = axis_iter.take();
        auto depth = axis_iters.size();
        if (axis_iter.at_buffer()) {
          // fill all lower level data_blocks values
          for (std::size_t i = depth; i < m_iter_params->size(); ++i) {
            AxisIter ai = AxisIter(iter_params(i), at_data);
            at_data = ai.at_data();
            data_blocks[ai.axis()] = ai.take(); 
          }
        }
        if (*inner_fileview_axis && axis == inner_fileview_axis->value())
          at_fileview_axis(axis);
        if (axis_iter.at_buffer())
          return;
        axis_iters.emplace_back(iter_params(depth), at_data);
      } else {
        axis_iters.pop_back(); 
      }
    }
  }

private:

  std::function<
    const std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned>&(
      const std::size_t&)> m_buffer_datatypes;

  std::function<
    const std::shared_ptr<const MPI_Datatype>&(
      const std::vector<finite_block_t>&)> m_fileview_datatypes;

  // std::ostringstream
  // show_iters(const std::deque<AxisIter>& iters) const {
  //   std::ostringstream result;
  //   std::for_each(
  //     std::begin(iters),
  //     std::end(iters),
  //     [&result](const auto& ai) {
  //       result << mscol_nickname(ai.params->axis) << ":" << ai.index << ";";
  //     });
  //   return result;
  // }
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
