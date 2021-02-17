#pragma once
#include "_support.h"
#include "CompactDiGraph.h"




template <class K=int, class V=NONE, class E=NONE>
class CompactGraph : public CompactDiGraph<K, V, E> {
  public:
  void addEdge(K u, K v, E d=E()) {
    CompactDiGraph<K, V, E>::addEdge(u, v, d);
    CompactDiGraph<K, V, E>::addEdge(v, u, d);
  }

  void removeEdge(K u, K v) {
    CompactDiGraph<K, V, E>::removeEdge(u, v);
    CompactDiGraph<K, V, E>::removeEdge(v, u);
  }
};
