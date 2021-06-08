#pragma once
#include <set>
#include <vector>
#include <functional>
#include "from.h"

using std::set;
using std::vector;
using std::hash;




// VERTICES
// --------

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


template <class G>
auto verticesByDegree(G& x) {
  auto a = vertices(x);
  sort(a.begin(), a.end(), [&](int u, int v) { return x.degree(u) < x.degree(v); });
  return a;
}




// VERTEX-HASH
// -----------

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




// VERTEX-DATA
// -----------

template <class G, class C>
auto vertexData(G& x, C&& ks) {
  using V = typename G::TVertex;
  vector<V> a;
  a.reserve(ks.size());
  for (auto u : ks)
    a.push_back(x.vertexData(u));
  return a;
}

template <class G>
auto vertexData(G& x) {
  return vertexData(x, x.vertices());
}




// VERTEX-CONTAINER
// ----------------

template <class G, class T, class C>
auto vertexContainer(G& x, vector<T>& vdata, C&& ks) {
  int i = 0;
  auto a = x.vertexContainer(T());
  for (auto u : ks)
    a[u] = vdata[i++];
  return a;
}

template <class G, class T>
auto vertexContainer(G& x, vector<T>& vdata) {
  return vertexContainer(x, vdata, x.vertices());
}




// SOURCE-OFFSETS
// --------------

template <class G, class C>
auto sourceOffsets(G& x, C&& ks) {
  int i = 0;
  vector<int> a;
  a.reserve(ks.size()+1);
  for (auto u : ks) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  return a;
}

template <class G>
auto sourceOffsets(G& x) {
  return sourceOffsets(x, x.vertices());
}
