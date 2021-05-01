#pragma once
#include "from.h"




template <class G, class H, class C>
void crossEdgesFor(H& a, G& x, C& c) {
  for (auto u : c) {
    for (auto v : x.edges(u))
      if (c.find(v)==c.end()) a.addEdge(u, v); // x.edgeData(u, v)
  }
}


template <class G, class H, class C>
void crossEdges(H& a, G& x, C& cs) {
  for (auto u : x.vertices())
    a.addVertex(u); // x.vertexData(u)
  for (auto& c : cs) {
    auto s = setFrom(c);
    crossEdgesFor(a, x, s);
  }
}

template <class G, class C>
auto crossEdges(G& x, C& cs) {
  G a; crossEdges(a, x, cs);
  return a;
}
