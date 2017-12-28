/* -*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil; -*- */
#ifndef ARRAY_INDEXER_H_
#define ARRAY_INDEXER_H_

#include <algorithm>
#include <memory>
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
  
  virtual std::size_t
  offset_of(const std::unordered_map<Columns, std::size_t>& index) const  = 0;

  virtual std::shared_ptr<ArrayIndexer>
  slice(const std::unordered_map<Columns, std::size_t>& fixed) const = 0;

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

  ArrayFullIndexer(
    ArrayOrder order,
    const std::vector<ColumnAxisBase<Columns> >& axes) {

    switch (order) {
    case ArrayOrder::row_major:
      std::for_each(
        std::begin(axes),
        std::end(axes),
        [this](const ColumnAxisBase<Columns>& c) { m_axes.push_back(c); });
      break;

    case ArrayOrder::column_major:
      std::for_each(
        std::rbegin(axes),
        std::rend(axes),
        [this](const ColumnAxisBase<Columns>& c) { m_axes.push_back(c); });
      break;
    }
  }

  std::size_t
  offset_of(const std::unordered_map<Columns, std::size_t>& index)
    const override {

    auto axes = m_axes.cbegin();
    std::size_t result = index.at(axes->id());
    ++axes;
    std::for_each(
      axes,
      m_axes.cend(),
      [&result, &index] (const ColumnAxisBase<Columns>& ax) mutable {
        result = index.at(ax.id()) + ax.length() * result;
      });
    return result;
  }

  std::shared_ptr<ArrayIndexer<Columns> >
  slice(const std::unordered_map<Columns, std::size_t>& fixed)
    const override {

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
};

template <typename Columns>
class ArraySliceIndexer
  : public ArrayIndexer<Columns> {

public:

  ArraySliceIndexer(
    const std::shared_ptr<ArrayFullIndexer<Columns> >& full,
    const std::unordered_map<Columns, std::size_t>& fixed)
    : m_full(full)
    , m_fixed(fixed) {
  }

  ArraySliceIndexer(
    const std::shared_ptr<ArrayFullIndexer<Columns> >& full,
    std::unordered_map<Columns, std::size_t>&& fixed)
    : m_full(full)
    , m_fixed(std::move(fixed)) {
  }

  std::size_t
  offset_of(const std::unordered_map<Columns, std::size_t>& index)
    const override {

    std::unordered_map<Columns, std::size_t> ix(index);
    std::for_each(
      std::begin(m_fixed),
      std::end(m_fixed),
      [&ix](const std::pair<Columns, std::size_t>& f) mutable {
        ix[f.first] = f.second;
      });
    return m_full->offset_of(ix);
  }

  std::shared_ptr<ArrayIndexer<Columns> >
  slice(const std::unordered_map<Columns, std::size_t>& fixed)
    const override {
    std::unordered_map<Columns, std::size_t> all_fixed(fixed);
    std::for_each(
      std::begin(m_fixed),
      std::end(m_fixed),
      [&all_fixed](const std::pair<Columns, std::size_t>& f)
      mutable {
        all_fixed[f.first] = f.second;
      });
    std::shared_ptr<ArraySliceIndexer<Columns> > result(
      new ArraySliceIndexer(m_full, std::move(all_fixed)));
    return std::static_pointer_cast<ArrayIndexer<Columns> >(result);
  }

private:

  std::shared_ptr<ArrayFullIndexer<Columns> > m_full;

  std::unordered_map<Columns, std::size_t> m_fixed;
};

} // end namespace mpims

#endif // ARRAY_INDEXER_H_
