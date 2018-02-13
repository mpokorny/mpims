/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef ARRAY_INDEXER_H_
#define ARRAY_INDEXER_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <ColumnAxis.h>

namespace mpims {

enum class ArrayOrder {
  row_major, column_major
};

template <typename Columns>
class ArraySliceIndexer;

template <typename Columns>
class ArrayFullIndexer;

template <typename Columns>
class ArrayIndexer {
public:

  typedef std::unordered_map<Columns, std::size_t> index;

  virtual std::optional<std::size_t>
  offset_of(const index&) const  = 0;

  virtual std::optional<std::size_t>
  offset_of_(const index&) const  = 0;

  virtual std::shared_ptr<ArrayIndexer>
  slice(const index& fixed) const = 0;

  static
  std::shared_ptr<ArrayIndexer>
  of(ArrayOrder order, const std::vector<ColumnAxisBase<Columns> >& axes) {
    std::shared_ptr<ArrayFullIndexer<Columns> > result(
      new ArrayFullIndexer<Columns>(order, axes));
    result->set_self_reference(result);
    return std::static_pointer_cast<ArrayIndexer<Columns> >(result);
  }
};

template <typename Columns>
class ArrayFullIndexer
  : public ArrayIndexer<Columns> {

  friend class ArrayIndexer<Columns>;

public:

  typedef typename ArrayIndexer<Columns>::index index;

  ArrayFullIndexer(
    ArrayOrder order,
    const std::vector<ColumnAxisBase<Columns> >& axes) {

    std::for_each(
      std::begin(axes),
      std::end(axes),
      [this](const ColumnAxisBase<Columns>& ax) {
        return m_axis_ids.insert(ax.id());
      });
    if (axes.size() != m_axis_ids.size())
      throw std::domain_error("Non-unique column id(s) specified by axes");

    switch (order) {
    case ArrayOrder::row_major:
      std::for_each(
        axes.begin(),
        axes.end(),
        [this](const ColumnAxisBase<Columns>& c) { m_axes.push_back(c); });
      break;

    case ArrayOrder::column_major:
      std::for_each(
        axes.rbegin(),
        axes.rend(),
        [this](const ColumnAxisBase<Columns>& c) { m_axes.push_back(c); });
      break;
    }
  }

  bool
  is_valid_index(const index& ix) const {

    std::set<Columns> cols;
    std::for_each(
      std::begin(ix),
      std::end(ix),
      [&cols](const std::pair<Columns, std::size_t>& i) {
        cols.insert(i.first);
      });
    return std::equal(
      std::begin(m_axis_ids), std::end(m_axis_ids), std::begin(cols));
  }

  std::optional<std::size_t>
  offset_of(const index& ix) const override {

    if (!is_valid_index(ix))
      throw std::domain_error("Invalid index axes");

    return offset_of_(ix);
  }

  std::optional<std::size_t>
  offset_of_(const index& ix) const override {

    auto axes = std::begin(m_axes);
    std::optional<std::size_t> result = ix.at(axes->id());
    ++axes;
    std::for_each(
      axes,
      std::end(m_axes),
      [&result, &ix] (const ColumnAxisBase<Columns>& ax) {
        if (result && !ax.is_unbounded())
          result = ix.at(ax.id()) + ax.length().value() * result.value();
        else
          result = std::nullopt;
      });
    return result;
  }

  std::shared_ptr<ArrayIndexer<Columns> >
  slice(const index& fixed) const override {

    std::shared_ptr<ArraySliceIndexer<Columns> > result(
      new ArraySliceIndexer<Columns>(m_self.lock(), fixed));
    return std::static_pointer_cast<ArrayIndexer<Columns> >(result);
  }

protected:

  std::weak_ptr<ArrayFullIndexer<Columns> > m_self;

  void
  set_self_reference(const std::shared_ptr<ArrayFullIndexer<Columns> >& ref) {
    m_self = std::weak_ptr<ArrayFullIndexer<Columns> >(ref);
  }

private:

  std::vector<ColumnAxisBase<Columns> > m_axes;

  std::set<Columns> m_axis_ids;
};

template <typename Columns>
class ArraySliceIndexer
  : public ArrayIndexer<Columns> {

public:

  typedef typename ArrayIndexer<Columns>::index index;

  ArraySliceIndexer(
    const std::shared_ptr<ArrayFullIndexer<Columns> >& full,
    const index& fixed)
    : m_full(full)
    , m_fixed(fixed) {
  }

  ArraySliceIndexer(
    const std::shared_ptr<ArrayFullIndexer<Columns> >& full,
    index&& fixed)
    : m_full(full)
    , m_fixed(std::move(fixed)) {
  }

  std::optional<std::size_t>
  offset_of(const index& ix) const override {

    return m_full->offset_of(*with_fixed(ix));
  }

  std::optional<std::size_t>
  offset_of_(const index& ix) const override {

    return m_full->offset_of_(*with_fixed(ix));
  }

  std::shared_ptr<ArrayIndexer<Columns> >
  slice(const index& fixed) const override {

    std::shared_ptr<ArraySliceIndexer<Columns> > result(
      new ArraySliceIndexer(m_full, std::move(*with_fixed(fixed))));
    return std::static_pointer_cast<ArrayIndexer<Columns> >(result);
  }

private:

  std::shared_ptr<ArrayFullIndexer<Columns> > m_full;

  index m_fixed;

  std::unique_ptr<index>
  with_fixed(const index& ix) const {

    std::unique_ptr<index> i(
      new std::unordered_map<Columns, std::size_t>(ix));
    std::for_each(
      std::begin(m_fixed),
      std::end(m_fixed),
      [&i](const std::pair<Columns, std::size_t>& f) mutable {
        (*i)[f.first] = f.second;
      });
    return i;
  }
};

} // end namespace mpims

#endif // ARRAY_INDEXER_H_
