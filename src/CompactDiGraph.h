#pragma once
#include <tuple>
#include <vector>
#include <iterator>
#include <algorithm>
#include "_support.h"
#include "filter.h"
#include "range.h"
#include "scan.h"
#include "transform.h"
#include "insert.h"
#include "erase.h"
#include <stdio.h>

using std::tuple;
using std::vector;
using std::min;
using std::count;
using std::lower_bound;
using std::transform;




template <class K=int, class V=NONE, class E=NONE>
class CompactDiGraph {
  vector<int> vto;
  vector<int> eto;
  vector<K> vkeys;
  vector<V> vdata;
  vector<E> edata;


  // Cutie helpers
  private:
  int n() { return vto.size()-1; }
  int m() { return eto.size(); }
  int estrt(int i) { return vto[min(i, n())]; }
  int estop(int i) { return vto[min(i+1, n())]; }
  auto vbgn() { return vto.begin(); }
  auto vend() { return vto.end(); }
  auto ebgn() { return eto.begin(); }
  auto eend() { return eto.end(); }
  auto vbgn(int i) { return vbgn()+i; }
  auto vend(int i) { return vbgn()+i+1; }
  auto ebgn(int i) { return ebgn()+estrt(i); }
  auto eend(int i) { return ebgn()+estop(i); }
  int escan(int i, int j) { return scan(ebgn(i), eend(i), j) - ebgn(); }
  int esrch(int i, int j) { int o = escan(i, j); return o == estop(i)? -1 : o; }
  bool hasv(int i) { return i < n(); }
  bool hase(int i, int j) { return hasv(i) && hasv(j) && esrch(i, j) >= 0; }

  int vi(K u) { return scanIndex(vkeys, u); }
  int ei(K u, K v) { return escan(vi(u), vi(v)); }
  K vk(int i) { return vkeys[i]; }

  int vscanl(K u) { return lower_bound(vkeys.begin(), vkeys.end(), u) - vkeys.begin(); }
  int escanl(int i, int j) { return lower_bound(ebgn(i), eend(i), j) - ebgn(); }
  void vadj(int i, int n) { transform(vbgn(i), vend(), vbgn(i), [&](int o) { return o+n; }); }
  void eadj(int i, int n) { transform(ebgn(), eend(), ebgn(), [&](int o) { return o<i? o:o+n; }); }

  void vins(int o, K u, V d) {
    insertAt(vto, o, vto[o]);
    insertAt(vkeys, o, u);
    insertAt(vdata, o, d);
    if (o != n()) eadj(o, 1);
  }

  void vdel(int o) {
    eraseAt(vto, o);
    eraseAt(vkeys, o);
    eraseAt(vdata, o);
    if (o != n()) eadj(o, -1);
  }

  void eins(int o, int j, E d) {
    insertAt(eto, o, j);
    insertAt(edata, o, d);
  }

  void edel(int o, int n) {
    eraseAt(eto, o, n);
    eraseAt(edata, o, n);
  }


  // For algorithms
  public:
  int edgeI(int o) { return eto[o]; }
  int edgesStartI(int i) { return estrt(i); }
  int edgesStopI(int i) { return estop(i); }
  int scanEdgeI(int i, int j) { return escan(i, j); }
  int searchEdgeI(int i, int j) { return esrch(i, j); }
  bool hasVertexI(int i) { return hasv(i); }
  bool hasEdgeI(int i, int j) { return hase(i, j); }

  auto verticesI() { return rangeIterable(n()); }
  auto edgesI(int i) { return transform(ebgn(i), eend(i), [=](int j) { return j; }); }
  auto inEdgesI(int j) { return filter(vto, [&](int i) { return esrch(i, j) >= 0; }); }
  int degreeI(int i) { return estop(i) - estrt(i); }
  int inDegreeI(int j) { return count(ebgn(), eend(), j); }
  K vertexKeyI(int i) { return vk(i); }

  V vertexDataI(int i) { return vdata[i]; }
  E edgeDataI(int o) { return edata[o]; }
  void setVertexDataI(int i, V d) { vdata[i] = d; }
  void setEdgeDataI(int o, E d) { edata[o] = d; }

  void addEdgeI(int i, int j, E d) {
    eins(escanl(i, j), j, d);
    vadj(i+1, 1);
  }

  void removeEdgeI(int i, int j) {
    edel(escan(i, j));
    vadj(i+1, -1);
  }

  void removeEdgesI(int i) {
    edel(estrt(i), degreeI(i));
    vadj(i+1, -degreeI(i));
  }

  void removeInEdgesI(int j) {
    for (int i=0, I=n(), r=0;; i++) {
      vto[i] -= r;
      if (i >= I) break;
      int o = esrch(i, j);
      if (o >= 0) { edel(o); r++; }
    }
  }

  void removeVertexI(int i) {
    removeEdgesI(i);
    removeInEdges(i);
    vdel(i);
  }


  // Arduino-like facade
  public:
  CompactDiGraph() { vto.push_back(0); }
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

  auto edges(K u) { return transform(edgesI(vi(u)), [&](int i) { return vk(i); }); }
  auto inEdges(K v) { return transform(inEdgesI(vi(v)), [&](int j) { return vk(j); }); }
  int degree(K u) { return degreeI(vi(u)); }
  int inDegree(K v) { return inDegreeI(vi(v)); }

  V vertexData(K u) { return hasVertex(u)? vertexDataI(vi(u)) : V(); }
  E edgeData(K u, K v) { return hasEdge(u, v)? edgeDataI(ei(u, v)) : E(); }
  void setVertexData(K u, V d) { if (hasVertex(u)) setVertexDataI(vi(u), d); }
  void setEdgeData(K u, K v, V d) { if (hasEdge(u, v)) setEdgeDataI(ei(u, v), d); }

  void removeEdge(K u, K v) { if (hasEdge(u, v)) removeEdgeI(vi(u), vi(v)); }
  void removeEdges(K u) { if (hasVertex(u)) removeEdgesI(vi(u)); }
  void removeInEdges(K v) { if (hasVertex(v)) removeInEdgesI(vi(v)); }
  void removeVertex(K u) { if (hasVertex(u)) removeVertexI(vi(u)); }
  void addVertex(K u, V d=V()) { if (!hasVertex(u)) vins(vscanl(u), u, d); }

  void addEdge(K u, K v, E d=E()) {
    if (hasEdge(u, v)) return;
    addVertex(u);
    addVertex(v);
    addEdgeI(vi(u), vi(v), d);
  }
};
