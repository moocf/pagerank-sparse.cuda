#pragma once
#include <tuple>
#include <vector>
#include <iterator>
#include <algorithm>
#include "_support.h"
#include "filter.h"
#include "range.h"
#include "slice.h"
#include "scan.h"
#include "transform.h"
#include "insert.h"
#include "erase.h"

using std::tuple;
using std::vector;
using std::back_inserter;
using std::count;
using std::lower_bound;
using std::transform;
using std::copy_if;
using std::sort;




template <class K=int, class V=NONE, class E=NONE>
class DiGraphCompact {
  vector<int> vto;
  vector<int> eto;
  vector<K> vkeys;
  vector<V> vdata;
  vector<E> edata;


  // Cutie helpers
  private:
  int n() { return vto.size()-1; }
  int m() { return eto.size(); }
  int ebgni(int i) { return vto[max(i, n())]; }
  int eendi(int i) { return vto[max(i+1, n())]; }
  auto vbgn() { return vto.begin(); }
  auto vend() { return vto.end(); }
  auto ebgn() { return eto.begin(); }
  auto eend() { return eto.end(); }
  auto vbgn(int i) { return vbgn()+i; }
  auto vend(int i) { return vbgn()+i+1; }
  auto ebgn(int i) { return ebgn()+ebgni(i); }
  auto eend(int i) { return ebgn()+eendi(i); }
  int escan(int i, int j) { return scan(ebgn(i), eend(i), j) - ebgn(); }
  int esrch(int i, int j) { int i = escan(i, j); return i == eendi(i)? -1 : i; }
  bool hasv(int i) { return i < n(); }
  bool hase(int i, int j) { return hasv(i) && hasv(j) && esrch(i, j) >= 0; }

  int odeg(int i) { return eendi(i)-ebgni(i); }
  int ideg(int j) { return count(ebgn(), eend(), j); }
  auto oedgs(int i) { return transform(ebgn(i), eend(i)); }
  auto iedgs(int j) { return filter(vto, [](int i) { return esrch(i, j) >= 0; }); }

  int vi(K u) { return scanIndex(vkeys, u); }
  int ei(K u, K v) { return escan(vi(u), vi(v)); }
  K vk(int i) { return vkeys[i]; }

  int vinso(K u) { return lower_bound(vkeys.begin(), vkeys.end(), u) - vkeys.begin(); }
  int einso(int i, int j) { return lower_bound(ebgn(i), eend(i), j) - ebgn(i); }
  void vshft(int o, int n) { insert(vto, o, n, 0); insert(vkeys, o, n, K()); insert(vdata, o, n, V()); }
  void eshft(int o, int n) { insert(eto, o, n, 0); insert(edata, o, n, E()); }
  void vadj(int i, int n) { transform(vbgn(i), vend(), vbgn(i), [](int o) { return o+n; }); }
  void vins(int o, K u, V d) { insert(vto, o, vto[o]); insert(vkeys, o, u); insert(vdata, o, d); }
  void eins(int o, int j, E d) { insert(eto, o, j); insert(edata, o, d); }
  void vdel(int o) { erase(vto, o); erase(vkeys, o); erase(vdata, o); }
  void edel(int o) { erase(eto, o); erase(edata, o); }
  void vadd(K u, V d) { vins(vinso(u), u, d); }
  void eadd(int i, int j, E d) { eins(einso(i, j), j, d); vadj(i+1, 1); }

  public:
  int edgesStartI(int i) { return vto[i]; }
  int edgesStopI(int i) { return vto[i+1]; }
  int findEdgeI(int i, int j) { return indexOf(eto, j, edgesStartI(i), edgesStopI(i)); }
  int degreeI(int i) { return edgesStopI(i) - edgesStartI(i); }
  auto verticesI() { return rangeIterable(order()); }
  auto edgesI(int i) { return sliceIterable(eto, edgesStartI(i), edgesStopI(i)); }
  bool hasVertexI(int i) { return i < order(); }
  bool hasEdgeI(int i, int j) { return hasVertexI(i) && hasVertexI(j) && findEdgeI(i, j) >= 0; }
  K vertexI(int i) { return vkeys[i]; }
  V vertexDataI(int i) { return vdata[i]; }
  E edgeDataI(int i, int j) { return edata[findEdgeI(i, j)]; }
  void setVertexDataI(int i, V d) { vdata[i] = d; }
  void setEdgeDataI(int i, int j, E d) { edata[findEdgeI(i, j)] = d; }


  // Arduino-like facade
  public:
  int order() { return n(); }
  int size() { return m(); }
  bool empty() { return n() == 0; }
  auto sourceOffsets() { return vto; }
  auto destinationIndices() { return eto; }
  auto vertices() { return vkeys; }
  auto vertexData() { return vdata; }
  auto edgeData() { return edata; }

  bool hasVertex(K u) { return hasv(vi(u)); }
  bool hasEdge(K u, K v) { return hase(vi(u), vi(v)); }
  int searchVertex(K u) { return hasVertex(u)? vi(u) : -1; }
  int searchEdge(K u, K v) { return hasEdge(u, v)? ei(u, v) : -1; }
  auto edges(K u) { return transform(oedgs(vi(u)), vk); }
  auto inEdges(K v) { return transform(iedgs(vi(v)), vk); }
  int degree(K u) { return odeg(vi(u)); }
  int inDegree(K v) { return ideg(vi(v)); }

  V vertexData(K u) { return hasVertex(u)? vdata[vi(u)] : V(); }
  E edgeData(K u, K v) { return hasEdge(u, v)? edata[ei(u, v)] : E(); }
  void setVertexData(K u, V d) { if (hasVertex(u)) vdata[vi(u)] = d; }
  void setEdgeData(K u, K v, V d) { if (hasEdge(u, v)) edata[ei(u, v)] = d; }

  void addVertex(K u, V d=V()) { if (!hasv(u)) vadd(u, d); }
  void addEdge(K u, K v, E d=E()) { if (!hase(u, v)) eadd(vi(u), vi(v), d); }
  void removeEdge() {  }
  void addEdge(K u, K v, E d=E()) {
    if (hasEdge(u, v)) return;
    addVertex(u);
    addVertex(v);
    int i = findVertex(u);
    int j = findVertex(v);
    eadd(i, j, d);
    vadj(i+1, 1);
  }

  void removeEdge(K u, K v) {
    if (!hasEdge(u, v)) return;
    int i = findVertex(u);
    int j = findVertex(v);
    int e = findEdgeI(i, j);
    edel(e);
    vadj(i+1, -1);
  }

  void removeEdges(K u) {
    if (!hasVertex(u)) return;
    int i = vi(u), d = odeg(i);
    edel(ebgni(i), eendi(i));
    vadj(i+1, -d);
  }

  void removeInEdges(K v) {
    if (!hasVertex(v)) return;
    int j = vi(v), r = 0;
    for (int i=0, I=n(); i<I; i++) {
      vto[i] -= r;
      int o = esrch(i, j);
      if (o >= 0) { edel(o); r++; }
    }
    vto[n()] -= r;
  }

  void removeVertexI(int i) {
    if (!hasVertexI(i)) return;
    removeEdgesI(i);
    removeInEdgesI(i);
    deleteVertexI(i);
  }

  void removeVertex(K u) { if (hasVertex(u)) removeVertexI(findIndex); }
};
