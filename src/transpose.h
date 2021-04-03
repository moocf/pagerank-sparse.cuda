#pragma once
#include "DiGraph.h"




// Reverses all links.
template <class G, class H>
auto& transpose(G& x, H& a) {
  for (auto u : x.vertices())
    a.addVertex(u, x.vertexData(u));
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(v, u, x.edgeData(u, v));
  }
  return a;
}

template <class G>
auto transpose(G& x) {
  G a; transpose(x, a);
  return a;
}




template <class G, class H>
auto& transposeWithDegree(G& x, H& a) {
  for (auto u : x.vertices())
    a.addVertex(u, x.degree(u));
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(v, u, x.edgeData(u, v));
  }
  return a;
}

template <class G>
auto transposeWithDegree(G& x) {
  using K = typename G::TKey;
  using E = typename G::TEdge;
  DiGraph<K, int, E> a; transposeWithDegree(x, a);
  return a;
}
