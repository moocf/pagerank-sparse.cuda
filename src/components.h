#pragma once
#include "DiGraph.h"
#include "dfs.h"
#include "dfsEnd.h"
#include "transpose.h"




// Finds Strongly Connected Components (SCC) using Kosaraju's algorithm.
vector<vector<int>> components(DiGraph& x) {
  // original dfs
  vector<bool> vis(x.span());
  vector<int> vs;
  for (int i=0, I=x.span(); i<I; i++)
    if (!vis[i]) dfsEndLoop(x, i, vis, vs);
  // transpose dfs
  vector<vector<int>> a;
  DiGraph* y = transpose(x);
  fill(vis.begin(), vis.end(), false);
  while (!vs.empty()) {
    int i = vs.pop_back();
    if (!vis[i]) a.push_back(dfs(*y, i));
  }
  return a;
}
