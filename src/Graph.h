#pragma once
#include "_support.h"
#include "DiGraph.h"




template <class K=int, class V=NONE, class E=NONE>
class Graph : public DiGraph<K, V, E> {
  public:
  void addEdge(K u, K v, E d=E()) {
    DiGraph<K, V, E>::addEdge(u, v, d);
    DiGraph<K, V, E>::addEdge(v, u, d);
  }

  void removeEdge(K u, K v) {
    DiGraph<K, V, E>::removeEdge(u, v);
    DiGraph<K, V, E>::removeEdge(v, u);
  }
};
