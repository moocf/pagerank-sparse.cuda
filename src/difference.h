#pragma once
#include "copy.h"




template <class G, class H>
void edgeDifference(G& a, G& x, H& y) {
  copy(a, x);
  for (auto u : y.vertices()) {
    for (auto v : y.edges(u))
      a.removeEdge(u, v);
  }
}

template <class G, class H>
auto edgeDifference(G& x, H& y) {
  G a; edgeDifference(a, x, y);
  return a;
}
