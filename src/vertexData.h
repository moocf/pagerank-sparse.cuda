#pragma once
#include <vector>

using std::vector;




template <class G, class C>
auto vertexData(G& x, C& ks) {
  using V = typename G::TVertex;
  vector<V> a;
  a.reserve(ks.size());
  for (auto u : ks)
    a.push_back(x.vertexData(u));
}

template <class G>
auto vertexData(G& x) {
  return vertexData(x, x.vertices());
}
