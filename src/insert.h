#pragma once
#include <vector>

using std::vector;




template <class T>
void insert(vector<T>& x, int i, T v) {
  if (i >= x.size()) x.push_back(v);
  else x.insert(x.begin()+i, v);
}

template <class T>
void insert(vector<T>& x, int i, int n, T v) {
  if (i >= x.size()) x.resize(x.size()+n);
  else x.insert(x.begin()+i, n, v);
}
