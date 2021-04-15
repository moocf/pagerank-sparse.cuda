#pragma once
#include <vector>
#include "edges.h"

using std::vector;




template <class G, class H, class K>
auto chainRoot(G& x, H& xt, K u) {
  K u0 = u;
  while (1) {
    K v = edge(xt, u);
    if (v == u0 || x.degree(v) != 1 || xt.degree(v) != 1) break;
    u = v;
  }
  return u;
}


template <class G, class H, class C, class K>
int hasChain(G& x, H& xt, C& vis, K u) {
  int n = 0;
  for (; n<2 && !vis[u] && x.degree(u) == 1 && xt.degree(u) == 1; n++)
    u = edge(x, u);
  return n >= 2;
}


template <class G, class H, class C, class K>
auto chainAt(G& x, H& xt, C& vis, K u) {
  vector<K> a;
  while (!vis[u] && x.degree(u) == 1 && xt.degree(u) == 1) {
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
    if (vis[u] || x.degree(u) != 1 || xt.degree(u) != 1) continue;
    u = chainRoot(x, xt, u);
    if (!hasChain(x, xt, vis, u)) continue;
    auto b = chainAt(x, xt, vis, u);
    a.push_back(b);
  }
  return a;
}
