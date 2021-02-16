#pragma once
#include <iterator>

using std::random_access_iterator_tag;
using std::iterator_traits;




template <class I, class F>
class TransformIterable {
  const I &ib, &ie;
  const F &fn;

  public:
  class Iterator {
    const F &fn;
    I it;

    public:
    using iterator_category = random_access_iterator_tag;
    using difference_type = typename iterator_traits<I>::difference_type;
    using value_type = decltype(fn(*it));
    using pointer = value_type*;
    using reference = value_type;

    Iterator(const F& fn, I it) : fn(fn), it(it) {}
    ITERATOR_DEREF(Iterator, fn(*it), fn(it[i]), NULL)
    ITERATOR_NEXTR(Iterator, ++it, --it)
    ITERATOR_ADVANCER(Iterator, i, it += i, it -= i)
    ITERATOR_ADVANCEP(Iterator, a, b, a.it+b)
    ITERATOR_ADVANCEN(Iterator, a, b, a.it-b)
    ITERATOR_COMPARISION(Iterator, a, b, a.it, b.it)
  };

  TransformIterable(const I& ib, const I& ie, F fn) : ib(ib), ie(ie), fn(fn) {}
  Iterator begin() { return Iterator(fn, ib); }
  Iterator end() { return Iterator(fn, ie); }
  Iterator rbegin() { return Iterator(fn, ie-1); }
  Iterator rend() { return Iterator(fn, ib-1); }
  ITERABLE_SIZE(ie-ib)
};


template <class I, class F>
auto transform(I ib, I ie, F fn) {
  return TransformIterable<I, F>(ib, ie, fn);
}


template <class I, class F>
auto transform(I i, F fn) {
  return TransformIterable<I, F>(i.begin(), i.end(), fn);
}
