#pragma once
#include "DiGraph.h"




// Reverses all links.
DiGraph* transpose(DiGraph& x) {
  DiGraph* a = new DiGraph();
  for (auto& i : x.vertices()) {
    for (auto& j : x.edges(i))
      a->addEdge(j, i);
  }
  return a;
}
