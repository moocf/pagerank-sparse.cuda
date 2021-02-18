#pragma once
#include <vector>
#include <iterator>
#include <algorithm>

using std::vector;
using std::iterator_traits;
using std::copy_if;
using std::back_inserter;



template <class I, class F>
auto filter(I ib, I ie, const F& fn) {
  using T = typename iterator_traits<I>::value_type;
  vector<T> a;
  copy_if(ib, ie, back_inserter(a), fn);
  return a;
}

template <class C, class F>
auto filter(C&& x, const F& fn) {
  return filter(x.begin(), x.end(), fn);
}


// #pragma once
// #include <iterator>

// using std::forward_iterator_tag;
// using std::iterator_traits;




// template <class I, class F>
// class FilterIterable {
//   const I ib, ie;
//   const F fn;

//   public:
//   class Iterator {
//     const F fn;
//     const I ie;
//     I it;

//     public:
//     using iterator_category = forward_iterator_tag;
//     using difference_type = typename iterator_traits<I>::difference_type;
//     using value_type = typename iterator_traits<I>::value_type;
//     using pointer = value_type*;
//     using reference = value_type;

//     Iterator(const F& fn, const I ie, I it) : fn(fn), ie(ie), it(it) { ++(*this); }
//     ITERATOR_DEREF(Iterator, *it, *it, NULL)
//     ITERATOR_NEXTP(Iterator, while(it!=ie && !fn(*it)) it++)
//     ITERATOR_ADVANCEP(Iterator, i, for (; i>=0; i--) ++(*this))
//     ITERATOR_COMPARISION(Iterator, a, b, a.it, b.it)
//   };

//   FilterIterable(I ib, I ie, const F& fn) : ib(ib), ie(ie), fn(fn) {}
//   Iterator begin() { return Iterator(fn, ie, ib); }
//   Iterator end() { return Iterator(fn, ie, ie); }
// };


// template <class I, class F>
// auto filter(I ib, I ie, const F& fn) {
//   return FilterIterable<I, F>(ib, ie, fn);
// }

// template <class C, class F>
// auto filter(C&& x, const F& fn) {
//   return filter(x.begin(), x.end(), fn);
// }
