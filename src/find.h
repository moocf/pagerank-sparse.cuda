#pragma once
#include <iterator>
#include <algorithm>

using std::find;
using std::distance;




template <class C, class T>
auto find(const C& x, const T& v) {
  return find(x.begin(), x.end(), v);
}

template <class I, class T>
int findAt(I ib, I ie, const T& v) {
  auto i = find(ib, ie, v);
  return i == ie? -1 : distance(ib, i);
}

template <class C, class T>
int findAt(const C& x, const T& v) {
  return findAt(x.begin(), x.end(), v);
}
