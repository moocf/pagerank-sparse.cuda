#pragma once
#include <tuple>
#include <vector>
#include <unordered_map>
#include <utility>
#include "_support.h"
#include "find.h"
#include "erase.h"
#include "range.h"
#include "count.h"
#include "transform.h"
#include "filter.h"

using std::tuple;
using std::vector;
using std::unordered_map;
using std::get;




template <class K=int, class V=NONE, class E=NONE>
class DiGraphBase {
  // Types
  public:
  using TKey    = K;
  using TVertex = V;
  using TEdge   = E;

  // Generate operations
  public:
  template <class T>
  auto createVertexData(T _) { return unordered_map<K, T>(); }

  template <class T>
  auto createEdgeData(T _)   { return unordered_map<tuple<K, K>, T>(); }
};




template <class K=int, class V=NONE, class E=NONE>
class DiGraph : public DiGraphBase<K, V, E> {
  vector<K> none;  // TODO: try removing this
  unordered_map<K, tuple<vector<K>, vector<E>, V>> ve;
  int M = 0;

  // Cute helpers
  private:
  int n() { return ve.size(); }
  auto& eto(K u)    { return get<0>(ve[u]); }
  auto& edata(K u)  { return get<1>(ve[u]); }
  auto& vdata(K u)  { return get<2>(ve[u]); }
  bool ex(K u, K v) { return find(eto(u), v) != eto(u).end(); }
  int  ei(K u, K v) { return find(eto(u), v) -  eto(u).begin(); }

  // Read operations
  public:
  auto& base()   { return *this; }
  int span()     { return n(); }
  int order()    { return n(); }
  int size()     { return M; }

  bool hasVertex(K u)    { return ve.find(u) != ve.end(); }
  bool hasEdge(K u, K v) { return hasVertex(u) && ex(u, v); }
  auto& edges(K u) { return hasVertex(u)? eto(u) : none; }
  int degree(K u)  { return hasVertex(u)? eto(u).size() : 0; }
  auto vertices()  { return transform(ve, [&](auto p) { return p.first; }); }
  auto inEdges(K v)   { return filter(ve, [&](auto p) { return ex(p.first, v); }); }
  int inDegree(K v)  { return countIf(ve, [&](auto p) { return ex(p.first, v); }); }

  V vertexData(K u)         { return hasVertex(u)? vdata(u) : V(); }
  void setVertexData(K u, V d) { if (hasVertex(u)) vdata(u) = d; }
  E edgeData(K u, K v)         { return hasEdge(u, v)? edata(u)[ei(u, v)] : E(); }
  void setEdgeData(K u, K v, E d) { if (hasEdge(u, v)) edata(u)[ei(u, v)] = d; }

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




template <class V, class E>
class DiGraph<int, V, E> : public DiGraphBase<int, V, E> {
  vector<int>  none;  // TODO: try removing this
  vector<bool> vex;
  vector<vector<int>> veto;
  vector<vector<E>>   edata;
  vector<V>           vdata;
  int N = 0, M = 0;

  // Cute helpers
  private:
  int s() { return veto.size(); }
  auto& eto(int u)      { return veto[u]; }
  bool ex(int u, int v) { return find(eto(u), v) != eto(u).end(); }
  int  ei(int u, int v) { return find(eto(u), v) -  eto(u).begin(); }

  // Read operations
  public:
  auto& base()   { return *this; }
  int span()     { return s(); }
  int order()    { return N; }
  int size()     { return M; }

  bool hasVertex(int u)      { return u < s() && vex[u]; }
  bool hasEdge(int u, int v) { return u < s() && ex(u, v); }
  auto& edges(int u)         { return u < s()? eto(u) : none; }
  int degree(int u)          { return u < s()? eto(u).size() : 0; }
  auto nonVertices()   { return filter(range(s()), [&](int u) { return !vex[u]; }); }
  auto vertices()      { return filter(range(s()), [&](int u) { return  vex[u]; }); }
  auto inEdges(int v)  { return filter(range(s()), [&](int u) { return ex(u, v); }); }
  int inDegree(int v) { return countIf(range(s()), [&](int u) { return ex(u, v); }); }

  V vertexData(int u)         { return hasVertex(u)? vdata[u] : V(); }
  void setVertexData(int u, V d) { if (hasVertex(u)) vdata[u] = d; }
  E edgeData(int u, int v)         { return hasEdge(u, v)? edata[u][ei(u, v)] : E(); }
  void setEdgeData(int u, int v, E d) { if (hasEdge(u, v)) edata[u][ei(u, v)] = d; }

  // Write operations
  public:
  void addVertex(int u, V d=V()) {
    if (hasVertex(u)) return;
    if (u >= s()) {
      vex.resize(u+1);
      veto.resize(u+1);
      edata.resize(u+1);
      vdata.resize(u+1);
    }
    vex[u] = true;
    vdata[u] = d;
    N++;
  }

  void addEdge(int u, int v, E d=E()) {
    if (hasEdge(u, v)) return;
    addVertex(u);
    addVertex(v);
    eto(u).push_back(v);
    edata[u].push_back(d);
    M++;
  }

  void removeEdge(int u, int v) {
    if (!hasEdge(u, v)) return;
    int o = ei(u, v);
    eraseAt(eto(u), o);
    eraseAt(edata[u], o);
    M--;
  }

  void removeEdges(int u) {
    if (!hasVertex(u)) return;
    M -= degree(u);
    eto(u).clear();
    edata[u].clear();
  }

  void removeInEdges(int v) {
    if (!hasVertex(v)) return;
    for (int u : inEdges(v))
      removeEdge(u, v);
  }

  void removeVertex(int u) {
    if (!hasVertex(u)) return;
    removeEdges(u);
    removeInEdges(u);
    vex[u] = false;
    N--;
  }

  // Generate operations
  public:
  template <class T>
  auto createVertexData(T _) { return vector<T>(span()); }
};
