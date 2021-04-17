#pragma once
#include <map>
#include <vector>
#include <utility>
#include "vertices.h"

using std::map;
using std::vector;
using std::pair;
using std::make_pair;
using std::move;




template <class G, class H, class C, class M>
void identicalSiblings(G& x, H& xt, C& vis, M& m) {
  using K = typename G::TKey;
  m.clear();
  for (K v : x.edges(u)) {
    if (vis[v]) continue;
    vis[v] = true;
    size_t h = vertexHash(xt, v);
    if (!m.count(h)) m[h] = make_pair({}, v);
    else m[h].first.push_back(v);
  }
  for (auto&& [_, p] : m) {
    if (p.first.empty()) continue;
    p.first.push_back(p.second);
    a.push_back(move(p.first));
  }
}


template <class G, class H>
auto identicals(G& x, H& xt) {
  using K = typename G::TKey;
  map<size_t, K> m;
  vector<vector<K>> a;
  auto vis = x.vertexContainer(bool());
  for (K u : x.vertices()) {
    if (x.degree(u)<2) continue;
    identicalSiblings(x, xt, vis, m);
  }
}
