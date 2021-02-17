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
auto filter(const C& x, const F& fn) {
  return filter(x.begin(), x.end(), fn);
}
