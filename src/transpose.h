#pragma once
#include "DiGraph.h"




// Reverses all links.
template <class K, class V, class E>
auto& transpose(DiGraph<K, V, E>& x) {
  auto &a = *new DiGraph<K, V, E>();
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(v, u, x.edgeData(u, v));
  }
  return a;
}
