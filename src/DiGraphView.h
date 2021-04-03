#pragma once




template <class G>
class DiGraphView {
  G& x;

  using K = typename G::TKey;
  using V = typename G::TVertex;
  using E = typename G::TEdge;
  using C = decltype(x.vertexContainer(bool()));
  C vex;

  // Types
  public:
  using TKey    = K;
  using TVertex = V;
  using TEdge   = E;

  // Read operations
  public:
  auto& base()   { return x; }
  int span()     { return x.span(); }
  int order()    { return x.order(); }
  int size()     { return x.size(); }

  bool hasVertex(int u)      { return x.hasVertex(u) && vex[u]; }
  bool hasEdge(int u, int v) { return x.hasEdge(u, v); }
  auto vertices()     { return filter(x.vertices(), [&](int u) { return hasVertex(u); }); }
  auto edges(int u)   { return filter(x.edges(u),   [&](int v) { return hasVertex(u); }); }
  auto inEdges(int v) { return filter(x.inEdges(v), [&](int u) { return hasVertex(v); }); }
  int degree(int u)   { return hasVertex(u)? x.degree(u)   : 0; }
  int inDegree(int v) { return hasVertex(v)? x.inDegree(v) : 0; }

  V vertexData(int u)      { return hasVertex(u)? x.vertexData(u) : V(); }
  E edgeData(int u, int v) { return hasEdge(u, v)? x.edgeData(u, v) : E(); }
  void setVertexData(int u, V d) {}
  void setEdgeData(int u, int v, E d) {}

  // Write operations
  public:
  void addVertex(int u, V d=V()) { if (x.hasVertex(u)) vex[u] = true; }
  void removeVertex(int u)       { if (x.hasVertex(u)) vex[u] = false; }
  void addEdge(int u, int v, E d=E()) {}
  void removeEdge(int u, int v) {}
  void removeEdges(int u) {}
  void removeInEdges(int v) {}

  // Generate operations
  public:
  template <class T>
  auto vertexContainer(T _) { return x.vertexContainer(); }

  template <class T>
  auto edgeContainer(T _)   { return x.edgeContainer(); }

  // Lifetime operations
  public:
  DiGraphView(G& _x) : x(_x) {}
};
