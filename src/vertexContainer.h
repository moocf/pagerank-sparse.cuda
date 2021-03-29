#pragma once
#include <vector>

using std::vector;




template <class G, class T, class C>
auto vertexContainer(G& x, vector<T>& vdata, C&& ks) {
  int i = 0;
  auto a = x.vertexContainer(T());
  for (auto u : ks)
    a[u] = vdata[i++];
  return a;
}

template <class G, class T>
auto vertexContainer(G& x, vector<T>& vdata) {
  return vertexContainer(x, vdata, x.vertices());
}
