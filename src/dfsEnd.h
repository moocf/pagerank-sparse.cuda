#pragma once
#include <vector>
#include "DiGraph.h"




// Traverses nodes in depth-first manner, listing on exit.
void dfsEndLoop(DiGraph& x, int i, vector<bool>& vis, vector<int>& a) {
  vis[i] = true;
  for (auto& j : x.edges(i))
    if (!vis[j]) dfsEndLoop(x, j, vis, a);
  a.push_back(i);
}


vector<int> dfsEnd(DiGraph& x, int i=0) {
  vector<bool> vis(x.span());
  vector<int> a;
  dfsEndLoop(x, i, vis, a);
  return a;
}
