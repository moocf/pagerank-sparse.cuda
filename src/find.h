#pragma once
#include <iterator>
#include <algorithm>

using std::find;
using std::distance;




template <class C, class T>
auto find(const C& x, const T& v) {
  return find(x.begin(), x.end(), v);
}

template <class C, class T>
int findAt(const C& x, const T& v) {
  auto i = find(x.begin(), x.end(), v);
  return i == x.end()? -1 : distance(x.begin(), i);
}
