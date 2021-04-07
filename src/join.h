#pragma once
#include <vector>

using std::vector;




template <class T>
auto join(vector<vector<T>>& xs) {
  vector<T> a;
  for (auto& x : xs)
    a.insert(a.end(), x.begin(), x.end());
  return a;
}


template <class T, class I>
auto joinFrom(vector<vector<T>>& xs, I&& is) {
  vector<T> a;
  for (int i : is)
    a.insert(a.end(), xs[i].begin(), xs[i].end());
  return a;
}
