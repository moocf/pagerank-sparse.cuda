#pragma once
#include <map>
#include <utility>
#include "_support.h"
#include "contains.h"
#include "transform.h"

using std::pair;
using std::map;



template <class K=int, class V=NONE, class E=NONE>
class DiGraphTemp {
  map<K, pair<V, map<K, E>>> x;
  int N = 0, M = 0;

  public:
  int order() { return N; }
  int size() { return M; }

  auto vertices() { return transform(x, [](auto&& p) { return p.first; }); }
  auto edges(K i) { return transform(x[i].second, [](auto&& p) { return p.second; }); }
  int degree(K i) { return x[i].second.size(); }

  bool hasVertex(K i) { return contains(x, i); }
  bool hasEdge(K i, K j) { return hasVertex(i) && hasVertex(j) && contains(x[i].second, j); }

  V vertexData(K i) { return x[i].first; }
  E edgeData(K i, K j) { return x[i].second[j]; }

  void addVertex(K i) {
    if (hasVertex(i)) return;
    x[i] = {};
    N++;
  }

  void removeVertex(K i) {
    if (!hasVertex(i)) return;
    x.erase(i);
    N--;
  }

  void addEdge(K i, K j) {
    if (hasEdge(i, j)) return;
    addVertex(i);
    addVertex(j);
    x[i].second[j] = {};
    M++;
  }

  void removeEdge(int i, int j) {
    if (!hasEdge(i, j)) return;
    x[i].second.erase(j);
    M--;
  }
};
