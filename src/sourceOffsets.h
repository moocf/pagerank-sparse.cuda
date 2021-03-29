#pragma once
#include <vector>

using std::vector;




template <class G, class C>
auto sourceOffsets(G& x, C&& ks) {
  int i = 0;
  vector<int> a;
  a.reserve(ks.size()+1);
  for (auto u : ks) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  return a;
}

template <class G>
auto sourceOffsets(G& x) {
  return sourceOffsets(x, x.vertices());
}
