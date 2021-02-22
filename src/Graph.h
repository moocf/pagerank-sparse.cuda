#pragma once




template <class G>
class Graph : public G {
  using K = typename G::TKey;
  using V = typename G::TVertex;
  using E = typename G::TEdge;

  // Read operations
  public:
  int size()   { return G::size()/2; }
  auto& base() { return (G&) (*this); }
  auto& root() { return G::root(); }

  // Write operations
  public:
  void addEdge(K u, K v, E d=E()) {
    G::addEdge(u, v, d);
    G::addEdge(v, u, d);
  }

  void removeEdge(K u, K v) {
    G::removeEdge(u, v);
    G::removeEdge(v, u);
  }
};
