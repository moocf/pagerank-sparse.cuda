#pragma once
#include <iterator>
#include <algorithm>

using std::lower_bound;
using std::distance;




template <class I, class T>
auto scan(I ib, I ie, const T& v) {
  I iv = lower_bound(ib, ie, v);
  return iv != ie && *iv == v? iv : ie;
}

template <class C, class T>
auto scan(const C& x, const T& v) {
  return scan(x.begin(), x.end(), v);
}

template <class C, class T>
int scanAt(const C& x, const T& v) {
  return distance(x.begin(), scan(x, v));
}
