#pragma once
#include <set>
#include <vector>
#include "from.h"
#include "contains.h"

using std::set;
using std::vector;




template <class G, class K>
auto identicalSiblings(G& xt, vector<K>& vs, int i, set<K>& s) {
  vector<K> a;
  K vi = vs[i];
  setFrom(s, xt.edges(vi));
  for (int j=i+1, J=vs.size(); j<J; j++) {
    auto vj = vs[j];
    if (s.size() != xt.degree(vj)) continue;
    if (contains(s, xt.edges(vj))) a.push_back(vj);
  }
  if (a.size() > 0) a.insert(0, vi);
  return a;
}


template <class G, class H>
auto inIdenticals(G& x, H& xt) {
  using K = typename G::TKey;
  vector<vector<K>> a;
  vector<K> vs;
  set<K> s;
  auto vis = x.vertexContainer(bool());
  for (auto u : x.vertices()) {
    if (x.degree(u) < 2) continue;

    vectorFrom(vs, x.edges(u));
    for (int i=0, I=vs.size(); i<I; i++) {
      if (vis[vs[i]]) continue;

      auto b = identicalSiblings(xt, vs, i, s);
      if (b.size() > 0) a.push_back(b);
      vis[vi] = true;
    }
  }
}
