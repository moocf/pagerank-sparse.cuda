#pragma once
#include <algorithm>
#include "add.h"
#include "fill.h"
#include "error.h"
#include "measure.h"
#include "pageRank.h"

using std::swap;



// Finds rank of nodes in graph.
template <class G, class C, class T>
void pageRankPushStep(C& a, C& r, G& x, T p) {
  int N = x.order();
  fill(a, (1-p)/N);
  for (int u : x.vertices()) {
    int d = x.degree(u);
    if (d > 0) addValueAt(a, p*r[u]/d, x.edges(u));
    else addValue(a, p*r[u]/N);
  }
}


template <class G, class C, class T>
auto& pageRankPushCore(float& m, C& a, C& r, G& x, T p, T E, int L) {
  T e0 = T();
  int N = x.order();
  fill(r, T(1)/N);
  int l = 0;
  for (; l<L; l++) {
    pageRankPushStep(a, r, x, p);
    T e1 = absError(a, r);
    if (e1 < E || e1 == e0) break;
    swap(a, r);
    e0 = e1;
  }
  m += l;
  return a;
}


template <class G, class H, class T=float>
auto pageRankPush(float& t, float& m, G& x, H& xt, PageRankOptions<T> o=PageRankOptions<T>()) {
  auto a = x.vertexContainer(T());
  auto r = x.vertexContainer(T()); m = 0;
  t = measureDuration([&]() { pageRankPushCore(m, a, r, x, o.damping, o.convergence, o.maxIterations); });
  fillAt(a, T(), x.nonVertices());
  return a;
}
