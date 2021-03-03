#pragma once
#include <vector>
#include <unordered_map>
#include "find.h"

using std::vector;
using std::unordered_map;




template <class G>
auto getVertexKeys(G& x) {
  using K = typename G::TKey;
  vector<K> a;
  a.reserve(x.order());
  for (auto u : x.vertices())
    a.push_back(u);
  return a;
}


template <class G>
auto getVertexData(G& x) {
  using V = typename G::TVertex;
  vector<V> a;
  a.reserve(x.order());
  for (auto u : x.vertices())
    a.push_back(x.vertexData(u));
  return a;
}


template <class G>
auto getEdgeData(G& x) {
  using E = typename G::TEdge;
  vector<E> a;
  a.reserve(x.size());
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.push_back(x.edgeData(u, v));
  }
  return a;
}


template <class G>
auto getSourceOffsets(G& x) {
  int i = 0;
  vector<int> a;
  a.reserve(x.order()+2);
  for (auto u : x.vertices()) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  a.push_back(i);
  return a;
}


template <class G>
auto getDestinationIndices(G& x) {
  using K = typename G::TKey;
  vector<int> a;
  unordered_map<K, int> id;
  auto ks = x.vertexKeys();
  for (int i=0, I=ks.size(); i<I; i++)
    id[ks[i]] = i;
  a.reserve(x.size());
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.push_back(id[v]);
  }
  return a;
}
