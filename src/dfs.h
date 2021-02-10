#pragma once
#include <vector>
#include "DiGraph.h"




// Traverses nodes in depth-first manner, listing on entry.
void dfsLoop(DiGraph& x, int i, vector<bool>& vis, vector<int>& a) {
  vis[i] = true;
  a.push_back(i);
  for (auto& j : x.edges(i))
    if (!vis[j]) dfsLoop(x, j, vis, a);
}


vector<int> dfs(DiGraph& x, int i=0) {
  vector<bool> vis(x.span());
  vector<int> a;
  dfsLoop(x, i, vis, a);
  return a;
}
