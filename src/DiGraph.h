#pragma once
#include <tuple>
#include <vector>
#include <unordered_map>
#include <utility>
#include "_support.h"
#include "find.h"
#include "transform.h"

using std::tuple;
using std::vector;
using std::unordered_map;
using std::get;




template <class K=int, class V=NONE, class E=NONE>
class DiGraph {
  unordered_map<K, tuple<vector<K>, vector<E>, V>> ve;
  int M = 0;

  // Cute helpers
  private:
  int n() { return ve.size(); }
  auto eto(K u) { return get<0>(ve[u]); }
  auto& edata(K u) { return get<1>(ve[u]); }
  auto& vdata(K u) { return get<2>(ve[u]); }
  int ei(K u, K v) { return findAt(eto(u), v); }

  // Read operations
  public:
  int span()   { return n(); }
  int order()  { return n(); }
  int size()   { return M; }
  bool empty() { return n() == 0; }

  bool hasVertex(K u)    { return ve.find(u) != ve.end(); }
  bool hasEdge(K u, K v) { return hasVertex(u) && ei(u, v) >= 0; }
  auto& edges(K u) { return eto(u); }
  int degree(K u)  { return eto(u).size(); }
  auto vertices()  { return transform(ve, [&](auto p) { return p.first; }); }
  auto inEdges(K v)   { return filter(ve, [&](auto p) { return ei(p.first, v) >= 0; }); }
  int inDegree(K v)  { return countIf(ve, [&](auto p) { return ei(p.first, v) >= 0; }); }

  V vertexData(K u)         { return hasVertex(u)? vdata(u) : V(); }
  void setVertexData(K u, V d) { if (hasVertex(u)) vdata(u) = d; }
  E edgeData(K u, K v)         { return hasEdge(u, v)? edata(u)[ei(u, v)] : E(); }
  void setEdgeData(K u, K v, V d) { if (hasEdge(u, v)) edata(u)[ei(u, v)] = d; }

  // Write operations
  public:
  void addVertex(K u, V d=V()) {
    if (hasVertex(u)) return;
    ve[u] = {{}, {}, V()};
  }

  void addEdge(K u, K v, E d=E()) {
    if (hasEdge(u, v)) return;
    addVertex(u);
    addVertex(v);
    eto(u).push_back(v);
    edata(u).push_back(d);
    M++;
  }

  void removeEdge(K u, K v) {
    if (!hasEdge(u, v)) return;
    int o = ei(u, v);
    eraseAt(edges(u), o);
    eraseAt(edata(u), o);
    M--;
  }

  void removeEdges(K u) {
    if (!hasVertex(u)) return;
    M -= degree(u);
    eto(u).clear();
    edata(u).clear();
  }

  void removeInEdges(K v) {
    if (!hasVertex(v)) return;
    for (auto&& u : inEdges(v))
      removeEdge(u, v);
  }

  void removeVertex(K u) {
    if (!hasVertex(u)) return;
    removeEdges(u);
    removeInEdges(u);
    ve.erase(u);
  }
};
