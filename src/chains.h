#pragma once
#include <vector>

using std::vector;




template <class G, class K, class C>
auto chainAt(G& x, K u, C& vis) {
  vector<K> a;
  while (!vis[u]) {
    vis[u] = true;
    if (x.degree(u) != 1) break;
    a.push_back(u);
    for (auto v : x.edges(u))
      u = v;
  }
  return a;
}


template <class G>
auto chains(G& x) {
  using K = typename G::TKey;
  vector<vector<K>> a;
  auto vis = x.vertexContainer(bool());
  for (auto u : x.vertices()) {
    auto b = chainAt(x, u, vis);
    if (b.size() > 0) a.push_back(b);
  }
  return a;
}
