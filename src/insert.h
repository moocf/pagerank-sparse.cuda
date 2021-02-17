#pragma once
#include <vector>

using std::vector;




template <class T>
void insertAt(vector<T>& x, int i, T v) {
  x.insert(x.begin()+i, v);
}

template <class T>
void insertAt(vector<T>& x, int i, int n, T v) {
  x.insert(x.begin()+i, n, v);
}
