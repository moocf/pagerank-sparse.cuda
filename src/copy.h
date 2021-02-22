#pragma once
#include "measureDuration.h"
#include <stdio.h>




template <class G, class H>
auto& copy(G& x, H& a) {
  for (auto u : x.vertices())
    a.addVertex(u, x.vertexData(u));
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(u, v, x.edgeData(u, v));
  }
  return a;
}
