#pragma once
#include <vector>
#include <iterator>
#include <algorithm>

using std::vector;
using std::copy_if;
using std::back_inserter;




template <class I, class F>
auto filter(I ib, I ie, F fn) {
  vector<decltype(*ib)> a;
  copy_if(ib, ie, back_inserter(a), fn);
  return a;
}

template <class I, class F>
auto filter(I i, F fn) {
  return filter(i.begin(), i.end(), fn);
}
