#pragma once
#include <vector>

using std::vector;




template <class G, class C>
auto edgeData(G& x, C&& ks) {
  using E = typename G::TEdge;
  vector<E> a;
  for (auto u : ks) {
    for (auto v : x.edges(u))
      a.push_back(x.edgeData(u, v));
  }
  return a;
}

template <class G>
auto edgeData(G& x) {
  return edgeData(x, x.vertices());
}
