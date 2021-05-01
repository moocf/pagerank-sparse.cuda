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


template <class T>
auto joinUntilSize(vector<vector<T>>& xs, int n) {
  vector<vector<T>> a;
  for (auto& x : xs) {
    if (a.empty() || n==0 || a.back().size()>=n) a.push_back(vector<T>());
    a.back().insert(a.back().end(), x.begin(), x.end());
  }
  return a;
}
