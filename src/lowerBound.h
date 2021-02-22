#pragma once
#include <iterator>
#include <algorithm>

using std::lower_bound;
using std::distance;




template <class I, class T>
auto lowerBound(I ib, I ie, const T& v) {
  return lower_bound(ib, ie, v);
}

template <class C, class T>
auto lowerBound(const C& x, const T& v) {
  return lowerBound(x.begin(), x.end(), v);
}

template <class C, class T>
int lowerBoundAt(const C& x, const T& v) {
  return distance(x.begin(), lowerBound(x, v));
}


template <class I, class T>
auto lowerBoundEqual(I ib, I ie, const T& v) {
  auto iv = lower_bound(ib, ie, v);
  return iv != ie && *iv == v? iv : ie;
}

template <class C, class T>
auto lowerBoundEqual(const C& x, const T& v) {
  return lowerBoundEqual(x.begin(), x.end(), v);
}

template <class C, class T>
int lowerBoundEqualAt(const C& x, const T& v) {
  return distance(x.begin(), lowerBoundEqual(x, v));
}
