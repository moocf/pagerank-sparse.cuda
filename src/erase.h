#pragma once
#include <vector>

using std::vector;




template <class T>
void erase(vector<T> x, int i) {
  x.erase(x.begin()+i);
}

template <class T>
void erase(vector<T> x, int i, int I) {
  x.erase(x.begin()+i, x.begin()+I);
}
