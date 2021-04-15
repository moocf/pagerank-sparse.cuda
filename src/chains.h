#pragma once
#include <vector>
#include "edges.h"

using std::vector;




template <class G, class H, class K>
auto chainRoot(G& x, H& xt, K u) {
  while (x.degree(u) == 1 && xt.degree(u) == 1)
    u = edge(xt, u);
  return u;
}


template <class G, class H, class K, class C>
auto chainAt(G& x, H& xt, K u, C& vis) {
  vector<K> a;
  while (x.degree(u) == 1 && xt.degree(u) == 1) {
    vis[u] = true;
    a.push_back(u);
    u = edge(x, u);
  }
  return a;
}


template <class G, class H>
auto chains(G& x, H& xt) {
  using K = typename G::TKey;
  vector<vector<K>> a;
  auto vis = x.vertexContainer(bool());
  for (auto u : x.vertices()) {
    u = chainRoot(x, xt, u);
    if (vis[u]) continue;
    vis[u] = true;
    for (auto v : x.edges(u)) {
      auto b = chainAt(x, xt, v, vis);
      if (b.size() < 2) continue;
      a.push_back(b);
    }
  }
  return a;
}
