#ifndef MEMO_FN_H_
#define MEMO_FN_H_

#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace mpims {

template <
  typename A,
  typename B,
  typename Hash = std::hash<A> >
class MemoFn {

public:

  template <
    typename F,
    typename = typename std::enable_if<
      std::is_convertible<std::invoke_result_t<F, A>, B>::value>::type>
  MemoFn(F&& f)
    : m_f(std::forward<F>(f))
    , m_cache(std::make_shared<std::unordered_map<A, B, Hash> >()) {
  }

  MemoFn(const MemoFn& f) {
    std::lock_guard<std::mutex> lock(f.m_mtx);
    m_f = f.m_f;
    m_mtx = f.m_mtx;
    m_cache = f.m_cache;
  }

  MemoFn(MemoFn&& f)
    : m_f(std::move(f).m_f)
    , m_mtx(std::move(f).m_mtx)
    , m_cache(std::move(f).m_cache) {
  }

  MemoFn&
  operator=(const MemoFn& rhs) {
    if (this != &rhs) {
      MemoFn tmp(rhs);
      swap(tmp);
    }
    return *this;
  }

  MemoFn&
  operator=(MemoFn&& rhs) {
    std::lock_guard<std::mutex> lock(*m_mtx);
    m_f = std::move(rhs).m_f;
    m_mtx = std::move(rhs).m_mtx;
    m_cache = std::move(rhs).m_cache;
  }
  
  const B&
  operator()(const A& a) const {
    std::lock_guard<std::mutex> lock(*m_mtx);
    if (m_cache->count(a) == 0)
      m_cache->insert(std::make_pair(a, m_f(a)));
    return (*m_cache)[a];
  }

protected:

  void
  swap(MemoFn& other) {
    using std::swap;
    std::lock(*m_mtx, *other.m_mtx);
    try {
      swap(m_f, other.m_f);
      swap(m_mtx, other.m_mtx);
      swap(m_cache, other.m_cache);
      m_mtx->unlock();
      other.m_mtx->unlock();
    } catch (...) {
      m_mtx->unlock();
      other.m_mtx->unlock();
      throw;
    }
  }

private:

  std::function<B(const A&)> m_f;

  mutable std::shared_ptr<std::mutex> m_mtx;

  mutable std::shared_ptr<std::unordered_map<A, B, Hash> > m_cache;
  
};

} // end namespace mpims

#endif // MEMO_FN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
