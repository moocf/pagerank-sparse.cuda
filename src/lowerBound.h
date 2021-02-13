#pragma once
#include <iterator>
#include <algorithm>

using std::lower_bound;
using std::distance;




template <class I, class T>
I lowerBound(I ib, I ie, const T& v) {
  return lower_bound(ib, ie, v);
}

template <class I, class T>
I lowerBound(I i, const T& v) {
  return lowerBound(i.begin(), i.end(), v);
}

template <class I, class T>
int lowerBoundIndex(I i, const T& v) {
  return distance(i.begin(), lowerBound(i, v));
}
