#pragma once
#include "pageRank.h"




// Finds rank of nodes in graph.
template <class G, class C, class T>
void pageRankPushStep(C& a, C& r, G& x, T p) {
  int N = x.order();
  fill(a, (1-p)/N);
  for (int u : x.vertices()) {
    int d = x.degree(u);
    if (d > 0) addAt(a, x.edges(u), p*r[u]/d);
    else add(a, p*r[u]/N);
  }
}


template <class G, class C, class T>
auto& pageRankPushCore(C& a, C& r, G& x, T p, T E) {
  T e0 = T();
  int N = x.order();
  fill(r, T(1)/N);
  while (1) {
    pageRankPushStep(a, r, x, p);
    T e1 = errorAbs(a, r);
    if (e1 < E || e1 == e0) break;
    swap(a, r);
    e0 = e1;
  }
  fillAt(a, x.nonVertices(), T());
  return a;
}


template <class G, class T>
auto pageRankPush(float& t, G& x, T p, T E) {
  auto a = x.vertexContainer(T());
  auto r = x.vertexContainer(T());
  t = measureDuration([&]() { pageRankPushCore(a, r, x, p, E); });
  return a;
}

template <class G, class T=float>
auto pageRankPush(float& t, G& x, PageRankOptions<T> o=PageRankOptions<T>()) {
  return pageRankPush(t, x, o.damping, o.convergence);
}
