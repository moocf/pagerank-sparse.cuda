#pragma once




// Reverses all links.
template <class G, class H>
auto& transpose(G& x, H& a) {
  for (auto u : x.vertices())
    a.addVertex(u, x.vertexData(u));
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(v, u, x.edgeData(u, v));
  }
  return a;
}




template <class G, class H>
auto& transposeWithDegree(G& x, H& a) {
  for (auto u : x.vertices())
    a.addVertex(u, x.degree(u));
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(v, u, x.edgeData(u, v));
  }
  return a;
}
