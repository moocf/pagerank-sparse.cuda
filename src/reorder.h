#pragma once
#include <algorithm>

using std::swap;




// Ref: https://stackoverflow.com/a/22183350/1413259
template <class T>
void reorder(vector<T>& x, vector<int> is) {
  for(int i=0, N=x.size(); i<N; i++) {
    while(is[i] != is[is[i]]) {
      swap(x[is[i]], x[is[is[i]]]);
      swap(  is[i],    is[is[i]]);
    }
  }
}
