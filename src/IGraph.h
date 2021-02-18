#pragma once
#include <vector>
#include "_support.h"

using std::vector;




template <class K=int, class V=NONE, class E=NONE>
class IGraph {
  public:
  int span()   { return 0; }
  int order()  { return 0; }
  int size()   { return 0; }
  bool empty() { return true; }

  bool hasVertex(K u)    { return false; }
  bool hasEdge(K u, K v) { return false; }
  const auto& vertices()   { return vector<K>(); }
  const auto& edges(K u)   { return vector<K>(); }
  const auto& inEdges(K v) { return vector<K>(); }
  int degree(K u)   { return 0; }
  int inDegree(K v) { return 0; }

  V vertexData(K u)    { return V(); }
  E edgeData(K u, K v) { return E(); }
  void setVertexData(K u, V d)    {}
  void setEdgeData(K u, K v, V d) {}

  void addVertex(K u, V d=V())    {}
  void addEdge(K u, K v, E d=E()) {}
  void removeEdge(K u, K v) {}
  void removeEdges(K u)     {}
  void removeInEdges(K v)   {}
  void removeVertex(K u)    {}
};
