#pragma once
#include <vector>
#include "edges.h"

using std::vector;




template <class G, class H, class K, class C>
void chainTraverse(vector<K>& a, G& x, H& xt, C& vis, K v) {
  while (!vis[v] && x.degree(v)==1 && xt.degree(v)==1) {
    vis[v] = true;
    a.push_back(v);
    v = edge(x, v);
  }
}


template <class G, class H>
auto chains(G& x, H& xt) {
  using K = typename G::TKey;
  vector<vector<K>> a;
  auto vis = x.vertexContainer(bool());
  for (K v : x.vertices()) {
    vector<K> b;
    if (vis[v] || xt.degree(v)!=1 || x.degree(v)!=1) continue;
    K u = edge(xt, v), w = edge(x, v);
    if ((vis[u] || xt.degree(u)!=1) && (vis[w] || x.degree(w)!=1)) continue;
    chainTraverse(b, xt, x, vis, u);
    chainTraverse(b, x, xt, vis, v);
    a.push_back(b);
  }
}
