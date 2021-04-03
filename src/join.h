#pragma once
#include <vector>

using std::vector;




template <class T, class C>
auto joinFrom(vector<vector<T>>& xs, C&& is) {
  vector<T> a;
  for (int i : is)
    a.insert(a.end(), xs[i].begin(), xs[i].end());
  return a;
}
