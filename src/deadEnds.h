#pragma once
#include <vector>

using std::vector;




template <class G>
auto deadEnds(G& x) {
  using K = typename G::TKey;
  vector<K> a;
  for (auto u : x.vertices())
    if (x.degree(u)==0) a.push_back(u);
  return a;
}




template <class G>
void loopDeadEnds(G& a) {
  for (auto u : a.vertices())
    if (a.degree(u)==0) a.addEdge(u, u);
}
