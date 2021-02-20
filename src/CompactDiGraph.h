#pragma once
#include <tuple>
#include <vector>
#include <iterator>
#include <algorithm>
#include "_support.h"
#include "filter.h"
#include "range.h"
#include "find.h"
#include "transform.h"
#include "lowerBound.h"
#include "insert.h"
#include "erase.h"
#include <stdio.h>

using std::tuple;
using std::vector;
using std::min;
using std::count;
using std::find;
using std::lower_bound;
using std::transform;




template <class V=NONE, class E=NONE>
class CompactDiGraphBase {
  vector<int> vto {0, 0};
  vector<int> eto;
  vector<V> vdata;
  vector<E> edata;

  // Types
  public:
  using TKey    = int;
  using TVertex = V;
  using TEdge   = E;

  // Cute helpers
  private:
  int n() { return vto.size()-2; }
  int m() { return eto.size(); }
  auto vbgn() { return vto.begin(); }
  auto vend() { return vto.end(); }
  auto ebgn() { return eto.begin(); }
  auto eend() { return eto.end(); }
  auto vbgn(int u) { return vbgn()+u; }
  auto vend(int u) { return vbgn()+u+1; }
  auto ebgni(int u) { return vto[u]; }
  auto eendi(int u) { return vto[u+1]; }
  auto ebgn(int u) { return eto.begin()+ebgni(u); }
  auto eend(int u) { return eto.begin()+eendi(u); }
  bool ex(int u, int v) { return find(ebgn(u), eend(u), v) != eend(u); }
  int  ei(int u, int v) { return find(ebgn(u), eend(u), v) -  ebgn(); }
  void vadj(int i, int n) { transformW(vbgn(i), vend(), [&](int o) { return o+n; }); }
  void eadj(int u, int n) { transformW(ebgn(), eend(),  [&](int x) { return x<u? x:x+n; }); }

  void eins(int o, int v, E d) {
    insertAt(eto, o, v);
    insertAt(edata, o, d);
  }

  void edel(int o, int n) {
    eraseAt(eto, o, n);
    eraseAt(edata, o, n);
  }

  // Read operations
  public:
  int span()   { return n(); }
  int order()  { return n(); }
  int size()   { return m(); }
  bool empty() { return n() == 0; }

  auto& base()               { return *this; }
  auto& sourceOffsets()      { return vto; }
  auto& destinationIndices() { return eto; }
  auto& vertexData()         { return vdata; }
  auto& edgeData()           { return edata; }

  bool hasVertex(int u)      { return u < n(); }
  bool hasEdge(int u, int v) { return ex(u, v); }
  auto edges(int u)          { return transform(ebgn(u), eend(u), IDENTITY); }
  int degree(int u)          { return eendi(u) - ebgni(u); }
  auto vertices()            { return range(n()); }
  auto inEdges(int v)  { return filter(range(n()), [&](int u) { return ex(u, v); }); }
  int inDegree(int v) { return countIf(range(n()), [&](int u) { return ex(u, v); }); }

  V vertexData(int u)            { return vdata[u]; }
  void setVertexData(int u, V d)        { vdata[u] = d; }
  E edgeData(int u, int v)       { return edata[ei(u, v)]; }
  void setEdgeData(int u, int v, E d)   { edata[ei(u, v)] = d; }

  // Write operations
  public:
  void addVertex(int u, V d=V()) {
    if (u != n()) eadj(u, 1);
    insertAt(vto, u, vto[u]);
    insertAt(vdata, u, d);
  }

  void addEdge(int u, int v, E d=E()) {
    if (hasEdge(u, v)) return;
    eins(ei(u, v), v, d);
    vadj(u+1, 1);
  }

  void removeEdge(int u, int v) {
    if (!hasEdge(u, v)) return;
    edel(ei(u, v), 1);
    vadj(u+1, -1);
  }

  void removeEdges(int u) {
    int d = degree(u);
    edel(ebgni(u), d);
    vadj(u+1, -d);
  }

  void removeInEdges(int v) {
    for (int u=0, U=n(), r=0;; u++) {
      vto[u] -= r;
      if (u >= U) break;
      if (hasEdge(u, v)) { edel(ei(u, v), 1); r++; }
    }
  }

  void removeVertex(int u) {
    removeEdges(u);
    removeInEdges(u);
    if (u != n()) eadj(u, -1);
    eraseAt(vto, u);
    eraseAt(vdata, u);
  }
};




template <class K=int, class V=NONE, class E=NONE>
class CompactDiGraph {
  CompactDiGraphBase<V, E> x;
  vector<K> vkeys;

  // Types
  public:
  using TKey    = K;
  using TVertex = V;
  using TEdge   = E;

  // Cute helpers
  private:
  int  vi(K u) { return find(vkeys, u) -  vkeys.begin(); }

  // Read operations
  public:
  int span()   { return x.span(); }
  int order()  { return x.order(); }
  int size()   { return x.size(); }
  bool empty() { return x.empty(); }

  auto& base()               { return x; }
  auto& sourceOffsets()      { return x.sourceOffsets(); }
  auto& destinationIndices() { return x.destinationIndices(); }
  auto& edgeData()           { return x.edgeData(); }
  auto& vertexData()         { return x.vertexData(); }
  auto& vertexKeys()         { return vkeys; }

  bool hasVertex(K u)    { return x.hasVertex(vi(u)); }
  bool hasEdge(K u, K v) { return hasVertex(u) && x.hasEdge(vi(u), vi(v)); }
  auto& vertices()       { return vkeys; }
  int degree(K u)   { return hasVertex(u)? x.degree(vi(u)) : 0; }
  int inDegree(K v) { return hasVertex(v)? x.inDegree(vi(v)) : 0; }
  auto edges(K u)     { return transform(x.edges(vi(u)), [&](int j) { return vkeys[j]; }); }
  auto inEdges(K v) { return transform(x.inEdges(vi(v)), [&](int i) { return vkeys[i]; }); }

  V vertexData(K u)         { return hasVertex(u)? x.vertexData(vi(u)) : V(); }
  void setVertexData(K u, V d) { if (hasVertex(u)) x.setVertexData(vi(u), d); }
  E edgeData(K u, K v)         { return hasEdge(u, v)? x.edgeData(vi(u), vi(v)) : E(); }
  void setEdgeData(K u, K v, E d) { if (hasEdge(u, v)) x.setEdgeData(vi(u), vi(v), d); }

  // Write operations
  public:
  void addVertex(K u, V d=V()) {
    if (hasVertex(u)) return;
    x.addVertex(vi(u), d);
    insertAt(vkeys, vi(u), u);
  }

  void addEdge(K u, K v, E d=E()) {
    if (hasEdge(u, v)) return;
    addVertex(u);
    addVertex(v);
    x.addEdge(vi(u), vi(v), d);
  }

  void removeEdge(K u, K v) {
    if (!hasEdge(u, v)) return;
    x.removeEdge(vi(u), vi(v));
  }

  void removeEdges(K u) {
    if (!hasVertex(u)) return;
    x.removeEdges(vi(u));
  }

  void removeInEdges(K v) {
    if (!hasVertex(v)) return;
    x.removeInEdges(vi(v));
  }

  void removeVertex(K u) {
    if (!hasVertex(u)) return;
    removeEdges(u);
    removeInEdges(u);
    x.removeVertex(vi(u));
    eraseAt(vkeys, vi(u));
  }
};
