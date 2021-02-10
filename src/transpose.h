#pragma once
#include "DiGraph.h"




// Reverses all links.
DiGraph& transpose(DiGraph& x) {
  DiGraph& a = *new DiGraph();
  for (int i=0, I=x.span(); i<I; i++) {
    if (!x.hasVertex(i)) continue;
    for (int j : x.edges(i))
      a.addEdge(j ,i);
  }
  return a;
}
