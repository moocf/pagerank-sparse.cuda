#pragma once
#include <vector>
#include "fill.h"
#include "dfs.h"

using std::vector;




// Finds Strongly Connected Components (SCC) using Kosaraju's algorithm.
template <class G>
auto components(G& x, G& y) {
  using K = typename G::TKey;
  // original dfs
  auto vis = x.vertexContainer(bool());
  vector<K> vs;
  for (auto u : x.vertices())
    if (!vis[u]) dfsEndLoop(x, u, vis, vs);
  // transpose dfs
  vector<vector<K>> a;
  fill(vis, false);
  while (!vs.empty()) {
    K u = vs.back();
    vs.pop_back();
    if (vis[u]) continue;
    a.push_back(vector<K>());
    dfsLoop(y, u, vis, a.back());
  }
  return a;
}
