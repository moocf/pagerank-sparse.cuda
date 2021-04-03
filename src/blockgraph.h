#pragma once
#include "components.h"




template <class G, class C>
auto& blockgraph(G& x, C& comps, G& a) {
  auto c = componentIds(x, comps);
  for (auto u : x.vertices()) {
    a.addVertex(c[u]);
    for (auto v : x.edges(u))
      if (c[u] != c[v]) a.addEdge(c[u], c[v]);
  }
  return a;
}

template <class G, class C>
auto blockgraph(G& x, C& comps) {
  G a; blockgraph(x, comps, a);
  return a;
}
