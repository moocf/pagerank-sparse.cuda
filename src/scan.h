#pragma once
#include <iterator>
#include <algorithm>

using std::lower_bound;
using std::distance;




template <class I, class T>
I scan(I ib, I ie, const T& v) {
  I iv = lower_bound(ib, ie, v);
  return *iv == v? iv : ie;
}

template <class I, class T>
I scan(I i, const T& v) {
  return scan(i.begin(), i.end(), v);
}

template <class I, class T>
int scanIndex(I i, const T& v) {
  return distance(i.begin(), scan(i, v));
}
