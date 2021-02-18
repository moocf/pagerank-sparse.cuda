#pragma once
#include <algorithm>

using std::count_if;




template <class I, class F>
int countIf(I ib, I ie, const F& fn) {
  return count_if(ib, ie, fn);
}

template <class C, class F>
int countIf(C&& x, const F& fn) {
  return count_if(x.begin(), x.end(), fn);
}
