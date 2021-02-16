#pragma once
#include <tuple>
#include <vector>
#include <map>
#include <utility>
#include "_support.h"
#include "scan.h"
#include "transform.h"

using std::tuple;
using std::vector;
using std::map;
using std::get;




template <class K=int, class V=NONE, class E=NONE>
class DiGraphTemp {
  map<K, tuple<vector<K>, vector<E>, V>> vto;
  int N = 0, M = 0;

  private:
  auto ebgn(K u) { return get<0>(vto[u]).begin(); }
  auto eend(K u) { return get<0>(vto[u]).end(); }
  auto ekbgn(K u) { return get<1>(vto[u]).begin(); }
  auto ekend(K u) { return get<1>(vto[u]).end(); }
  auto& vdata(K u) { return get<2>(vto[u]); }
  auto& edata(K u, K v) { }
  int escan(K u, K v) { return scan(ebgn(u), eend(u), v) - ebgn(u); }
  int esrch(K u, K v) { int i = escan(u, v); return i == estop(i)? -1 : i; }

  public:
  int order() { return N; }
  int size() { return M; }
  bool empty() { return N == 0; }

  bool hasVertex(K u) { return vto.find(u) != vto.end(); }
  bool hasEdge(K u, K v) { return hasVertex(u) && hasVertex(v) && esrch(u, v) >= 0; }

  auto vertices() { return transform(vto, [](auto p) { return p.first; }); }
  auto& edges(K u) { return get<0>(vto[u]); }
  int degree(K u) { return edges(u).size(); }

  auto inEdges(K v) { return filter(vto, [=](auto p) { return esrch(p.first, v) >= 0; }); }

  int inDegree(K v) {
    int a = 0;
    for (auto&& [u, e] : vto)
      if (esrch(u, v) >= 0) a++;
    return a;
  }

  V vertexData(K u) { return hasVertex(u)? vdata(u) : V(); }
  E edgeData(K u, K v) { return hasEdge(u, v)? edata(u, v) : E(); }
  void setVertexData(K u, V d) { if (hasVertex(u)) vdata(u) = d; }
  void setEdgeData(K u, K v, V d) { if (hasEdge(u, v)) edata(u, v) = d; }

  void addVertex(K u, V d=V()) {
    if (hasVertex(u)) return;
    vto[u] = {{}, {}, V()};
    N++;
  }

  void addEdge(K u, K v, E d=E()) {
    if (hasEdge(u, v)) return;
    edges(u).push_back({v, d});
    M++;
  }

  void removeEdge(K u, K v) {
    if (!hasEdge(u, v)) return;
    erase(edges(u), escan(u, v));
    M--;
  }

  void removeEdges(K u) {
    if (!hasVertex(u)) return;
    M -= edges(u).size();
    edges(u).clear();
  }

  void removeInEdges(K v) {
    if (!hasVertex(v)) return;
    for (auto&& [u, e] : vto)
      if (hasEdge(u, v)) removeEdge(u, v);
  }

  void removeVertex(K u) {
    if (!hasVertex(u)) return;
    removeEdges(u);
    removeInEdges(u);
    vto.erase(u);
    N--;
  }
};
