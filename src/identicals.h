#pragma once




template <class G, class H>
auto inIdenticals(G& x, H& xt) {
  auto vis = x.vertexContainer(bool());
  for (auto u : x.vertices()) {
    if (x.degree(u) < 2) {  }
    for (auto v : x.edges(u)) {

    }
  }
}
