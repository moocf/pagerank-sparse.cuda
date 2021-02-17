#pragma once
#include <vector>

using std::vector;




template <class T>
void eraseAt(vector<T>& x, int i) {
  x.erase(x.begin()+i);
}

template <class T>
void eraseAt(vector<T>& x, int i, int n) {
  x.erase(x.begin()+i, x.begin()+i+n);
}
