#pragma once
#include <set>
#include <vector>
#include <functional>
#include "from.h"

using std::set;
using std::vector;
using std::hash;




template <class G>
auto vertices(G& x) {
  using K = typename G::TKey;
  vector<K> a;
  a.reserve(x.order());
  for (auto u : x.vertices())
    a.push_back(u);
  return a;
}


template <class G, class F>
auto verticesBy(G& x, F fm) {
  auto a = vertices(x);
  sort(a.begin(), a.end(), [&](auto u, auto v) {
    return fm(u) < fm(v);
  });
  return a;
}


template <class G, class S, class K>
size_t vertexHash(G& x, S& es, K u) {
  size_t a = 0;
  setFrom(es, x.edges(u));
  for (K v : es)
    a ^= hash<K>{}(v) + 0x9e3779b9 + (a<<6) + (a>>2); // from boost::hash_combine
  return a;
}

template <class G, class K>
size_t vertexHash(G& x, K u) {
  set<K> es;
  return vertexHash(x, es, u);
}
