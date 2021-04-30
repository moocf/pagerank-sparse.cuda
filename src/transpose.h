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




template <class G, class H>
void transposeForNvgraph(H& a, G& x) {
  using V = typename H::TVertex;
  using E = typename H::TEdge;
  for (auto u : x.vertices())
    a.addVertex(u, x.degree(u) == 0? V(1) : V(0));
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(v, u, E(1)/x.degree(u));
  }
}

template <class G>
auto transposeForNvgraph(G& x) {
  using K = typename G::TKey;
  DiGraph<K, float, float> a; transposeForNvgraph(a, x);
  return a;
}
