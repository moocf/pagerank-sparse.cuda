#pragma once
#include "DiGraphView.h"




template <class G, class C>
auto subgraph(G& x, C& ks) {
  DiGraphView<G> a(x);
  for (auto u : ks)
    a.addVertex(u);
  return a;
}
