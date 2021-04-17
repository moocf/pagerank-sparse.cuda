#pragma once
#include <set>
#include <map>
#include <vector>
#include <utility>
#include "vertices.h"

using std::set;
using std::map;
using std::vector;
using std::pair;
using std::make_pair;
using std::move;




template <class G, class H>
auto inIdenticals(G& x, H& xt) {
  using K = typename G::TKey;
  set<K> es; map<size_t, pair<vector<K>, K>> m;
  vector<vector<K>> a;
  auto vis = x.vertexContainer(bool());
  for (K u : x.vertices()) {
    if (x.degree(u)<2) continue;
    m.clear();
    for (K v : x.edges(u)) {
      if (vis[v]) continue;
      vis[v] = true;
      size_t h = vertexHash(xt, es, v);
      if (!m.count(h)) m[h] = make_pair(vector<K>(), v);
      else m[h].first.push_back(v);
    }
    for (auto&& [_, p] : m) {
      if (p.first.empty()) continue;
      p.first.insert(p.first.begin(), p.second);
      a.push_back(move(p.first));
    }
  }
  return a;
}
