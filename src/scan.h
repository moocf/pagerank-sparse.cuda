#pragma once
#include <iterator>
#include <algorithm>

using std::lower_bound;
using std::distance;




template <class I, class T>
auto scan(const I& ib, const I& ie, const T& v) {
  const I& iv = lower_bound(ib, ie, v);
  return iv != ie && *iv == v? iv : ie;
}

template <class I, class T>
auto scan(const I& i, const T& v) {
  return scan(i.begin(), i.end(), v);
}

template <class I, class T>
int scanIndex(const I& i, const T& v) {
  return distance(i.begin(), scan(i, v));
}
