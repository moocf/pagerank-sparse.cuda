#pragma once
#include <stdio.h>




template <class G, class H>
auto& copy(G& x, H& a) {
  printf("copy: vertices ...\n");
  for (auto u : x.vertices())
    a.addVertex(u, x.vertexData(u));
  printf("copy: edges ...\n");
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(u, v, x.edgeData(u, v));
  }
  printf("copy: done\n");
  return a;
}
