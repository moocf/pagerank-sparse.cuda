#pragma once
#include <tuple>
#include <vector>
#include <utility>
#include "_support.h"
#include "IGraph.h"
#include "count.h"
#include "find.h"
#include "filter.h"
#include "range.h"
#include "transform.h"

using std::tuple;
using std::vector;
using std::get;




template <class K=int, class V=NONE, class E=NONE>
class DiGraph : public IGraph<K, V, E> {
  vector<bool> vex;
  vector<vector<int>> eto;
  vector<vector<E>> edata;
  vector<K> vkeys;
  vector<V> vdata;
  vector<K> none;
  int N = 0, M = 0;


  // Cute helpers
  int s() { return eto.size(); }
  int ei(int u, K v) { return findAt(eto[u], v); }

  // Arduino-like facade
  public:
  int span()   { return s(); }
  int order()  { return N; }
  int size()   { return M; }
  bool empty() { return N == 0; }

  bool hasVertex(K u)    { return u < s() && vex[u]; }
  bool hasEdge(K u, K v) { return u < s() && ei(u, v) >= 0; }
  auto& edges(K u)       { return u < s()? eto[u] : none; }
  int degree(K u)        { return u < s()? eto[u].size() : 0; }
  auto vertices()   { return filter(range(s()), [&](int u) { return vex[u]; }); }
  auto inEdges(K v) { return filter(range(s()), [&](int u) { return hasEdge(u, v); }); }
  int inDegree(K v) { return countIf(range(s()), [&](int u) { return hasEdge(u, v); }); }

  V vertexData(K u)         { return hasVertex(u)? vdata[u] : V(); }
  void setVertexData(K u, V d) { if (hasVertex(u)) vdata[u] = d; }
  E edgeData(K u, K v)         { return hasEdge(u, v)? edata[u][ei(u, v)] : E(); }
  void setEdgeData(K u, K v, V d) { if (hasEdge(u, v)) edata[u][ei(u, v)] = d; }

  void addVertex(K u, V d=V()) {
    if (hasVertex(u)) return;
    if (u >= s()) {
      vex.resize(u+1);
      eto.resize(u+1);
      edata.resize(u+1);
      vdata.resize(u+1);
    }
    vex[u] = true;
    vdata[u] = d;
    N++;
  }

  void addEdge(K u, K v, E d=E()) {
    if (hasEdge(u, v)) return;
    addVertex(u);
    addVertex(v);
    eto[u].push_back(v);
    edata[u].push_back(d);
    M++;
  }

  void removeEdge(K u, K v) {
    if (!hasEdge(u, v)) return;
    int o = findAt(u, v);
    eraseAt(eto[u], o);
    eraseAt(edata[u], o);
    M--;
  }

  void removeEdges(K u) {
    if (!hasVertex(u)) return;
    M -= degree(u);
    eto[u].clear();
    edata[u].clear();
  }

  void removeInEdges(K v) {
    if (!hasVertex(v)) return;
    for (int u=0, U=s(); u<U; u++)
      removeEdge(u, v);
  }

  void removeVertex(K u) {
    if (!hasVertex(u)) return;
    removeEdges(u);
    removeInEdges(u);
    vex[u] = false;
    N--;
  }
};
