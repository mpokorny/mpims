#ifndef TRAVERSAL_STATE_H_
#define TRAVERSAL_STATE_H_

#include <deque>
#include <numeric>
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
    MPI_Comm comm,
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
    m_num_iterations = max_iterations(comm, m_iter_params);

    // initialize data_blocks values
    bool at_data = true;
    for (std::size_t i = 0; i < m_iter_params->size(); ++i) {
      AxisIter ai =
        AxisIter(this->iter_params(i), m_num_iterations[i], at_data);
      at_data = ai.at_data();
      data_blocks[ai.axis()] = ai.take();
    }

    axis_iters.emplace_back(this->iter_params(0), m_num_iterations[0], true);
    if (axis_iters.front().axis() == outer_ms_axis)
      initial_outer_length = outer_ms_length;
    cont = true;
  }

  TraversalState(const TraversalState& other)
    : cont(other.cont)
    , global_eof(other.global_eof)
    , m_iter_params(other.m_iter_params)
    , data_blocks(other.data_blocks)
    , axis_iters(other.axis_iters)
    , outer_full_array_axis(other.outer_full_array_axis)
    , initial_outer_length(other.initial_outer_length)
    , m_buffer_datatypes(other.m_buffer_datatypes)
    , m_fileview_datatypes(other.m_fileview_datatypes)
    , m_num_iterations(other.m_num_iterations) {
  }

  TraversalState(TraversalState&& other)
    : cont(other.cont)
    , global_eof(other.global_eof)
    , m_iter_params(std::move(other).m_iter_params)
    , data_blocks(std::move(other).data_blocks)
    , axis_iters(std::move(other).axis_iters)
    , outer_full_array_axis(other.outer_full_array_axis)
    , initial_outer_length(other.initial_outer_length)
    , m_buffer_datatypes(other.m_buffer_datatypes)
    , m_fileview_datatypes(other.m_fileview_datatypes)
    , m_num_iterations(std::move(other).m_num_iterations) {
  }

  TraversalState&
  operator=(const TraversalState& rhs) {
    if (this != &rhs) {
      TraversalState temp(rhs);
      swap(temp);
    }
    return *this;
  }

  TraversalState&
  operator=(TraversalState&& rhs) {
    cont = rhs.cont;
    global_eof = rhs.global_eof;
    m_iter_params = std::move(rhs).m_iter_params;
    data_blocks = std::move(rhs).data_blocks;
    axis_iters = std::move(rhs).axis_iters;
    outer_full_array_axis = rhs.outer_full_array_axis;
    initial_outer_length = rhs.initial_outer_length;
    m_buffer_datatypes = rhs.m_buffer_datatypes;
    m_fileview_datatypes = rhs.m_fileview_datatypes;
    m_num_iterations = std::move(rhs).m_num_iterations;
    return *this;
  }

  //EOF condition
  bool
  eof() const {
    if (axis_iters.empty() || global_eof)
      return true;
    if (axis_iters.size() != 1)
      return false;
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
      std::size_t count =
        std::accumulate(
          std::begin(*blks),
          std::end(*blks),
          0,
          [](auto& acc, auto& blk) { return acc + std::get<1>(blk); });
      return m_buffer_datatypes(count);
    } else {
      return zilch;
    }
  }

  const std::shared_ptr<const MPI_Datatype>&
  fileview_datatype(MSColumns fileview_axis)
    const {
    // find data_blocks at fileview axis
    std::vector<finite_block_t> blks;
    const std::vector<finite_block_t>* blks_p = nullptr;
    for (std::size_t i = 0; i < axis_iters.size() && !blks_p; ++i) {
      if (axis_iters[i].axis() == fileview_axis)
        blks_p = &data_blocks.at(axis_iters[i].axis());
    }
    assert(blks_p);

    if (blks_p->size() > 0) {
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
  advance_to_next_buffer(MSColumns inner_fileview_axis, F at_fileview_axis) {

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
            AxisIter ai =
              AxisIter(iter_params(i), m_num_iterations[i], at_data);
            at_data = ai.at_data();
            data_blocks[ai.axis()] = ai.take();
          }
        }
        if (axis == inner_fileview_axis)
          at_fileview_axis();
        if (axis_iter.at_buffer())
          return;
        axis_iters.emplace_back(
          iter_params(depth),
          m_num_iterations[depth],
          at_data);
      } else {
        axis_iters.pop_back();
      }
    }
  }

  // continuation condition flag (to allow breaking out from iteration)
  bool cont;

  bool global_eof;

  std::shared_ptr<const std::vector<IterParams> > m_iter_params;

  std::unordered_map<MSColumns, std::vector<finite_block_t> > data_blocks;

  // stack of AxisIters to maintain axis iteration indexes
  std::deque<AxisIter> axis_iters;

  // the outermost axis at which data can fully fit into a buffer
  MSColumns outer_full_array_axis;

  std::optional<std::size_t> initial_outer_length;

protected:

  void
  swap(TraversalState& other) {
    using std::swap;
    swap(cont, other.cont);
    swap(global_eof, other.global_eof);
    swap(m_iter_params, other.m_iter_params);
    swap(data_blocks, other.data_blocks);
    swap(axis_iters, other.axis_iters);
    swap(outer_full_array_axis, other.outer_full_array_axis);
    swap(initial_outer_length, other.initial_outer_length);
    swap(m_buffer_datatypes, other.m_buffer_datatypes);
    swap(m_fileview_datatypes, other.m_fileview_datatypes);
    swap(m_num_iterations, other.m_num_iterations);
  }

private:

  std::function<
    const std::tuple<std::shared_ptr<const MPI_Datatype>, unsigned>&(
      const std::size_t&)> m_buffer_datatypes;

  std::function<
    const std::shared_ptr<const MPI_Datatype>&(
      const std::vector<finite_block_t>&)> m_fileview_datatypes;

  std::vector<std::optional<std::size_t> > m_num_iterations;

  static std::vector<std::optional<std::size_t> >
  max_iterations(
    MPI_Comm comm,
    const std::shared_ptr<const std::vector<IterParams> >& iter_params) {

    // type bool won't work for inf_nit, since vector<bool> is specialized such
    // that accessing the data for MPI communication won't work
    std::vector<unsigned char> inf_nit;
    std::vector<std::size_t> nit;
    inf_nit.reserve(iter_params->size());
    nit.reserve(iter_params->size());
    std::for_each(
      std::begin(*iter_params),
      std::end(*iter_params),
      [&](auto& ip) {
        auto num_iterations = ip.num_total_iterations();
        if (num_iterations) {
          inf_nit.emplace_back(false);
          nit.emplace_back(num_iterations.value());
        } else {
          inf_nit.emplace_back(true);
          nit.emplace_back(0);
        }
      });
    MPI_Allreduce(
      MPI_IN_PLACE,
      inf_nit.data(),
      inf_nit.size(),
      MPI_UNSIGNED_CHAR,
      MPI_LOR,
      comm);
    MPI_Allreduce(
      MPI_IN_PLACE,
      nit.data(),
      nit.size(),
      MPIMS_SIZE_T,
      MPI_MAX,
      comm);
    std::vector<std::optional<std::size_t> > result;
    result.reserve(inf_nit.size());
    for (std::size_t i = 0; i < inf_nit.size(); ++i) {
      if (inf_nit[i])
        result.emplace_back(std::nullopt);
      else
        result.emplace_back(nit[i]);
    }
    return result;
  }

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
