#pragma once
#include <vector>
#include "fill.h"
#include "dfs.h"

using std::vector;




// Finds Strongly Connected Components (SCC) using Kosaraju's algorithm.
template <class G, class H>
auto components(G& x, H& xt, int n0=0) {
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
    if (a.empty() || !n0 || n0 && a.back().size() >= n0)
      a.push_back(vector<K>());
    dfsLoop(xt, u, vis, a.back());
  }
  return a;
}


template <class G, class C>
auto componentIds(G& x, C& comps) {
  int i = 0;
  auto a = x.vertexContainer(int());
  for (auto& c : comps) {
    for (auto u : c)
      a[u] = i;
    i++;
  }
  return a;
}
