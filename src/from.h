#pragma once
#include <set>
#include <vector>
#include <iterator>

using std::set;
using std::vector;
using std::iterator_traits;




template <class I, class C>
void setFrom(C& a, I&& x) {
  a.clear();
  a.insert(x.begin(), x.end());
}

template <class I>
auto setFrom(I&& x) {
  using T = typename std::iterator_traits<I>::value_type;
  set<T> a; setFrom(a, x);
  return a;
}




template <class I, class C>
void vectorFrom(C& a, I&& x) {
  a.clear();
  a.reserve(x.size());
  for (auto v : x)
    a.push_back(v);
}

template <class I>
auto vectorFrom(I&& x) {
  using T = typename std::iterator_traits<I>::value_type;
  vector<T> a; vectorFrom(a, x);
  return a;
}
