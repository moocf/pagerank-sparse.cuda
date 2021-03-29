#pragma once
#include <vector>

using std::vector;




template <class G>
auto vertices(G& x) {
  using K = typename G::TKey;
  vector<K> a;
  a.reserve(x.order());
  for (auto u : x.vertices())
    a.push_back(u);
  return a;
}


template <class G, class F>
auto verticesBy(G& x, F fm) {
  auto a = vertices(x);
  sort(a.begin(), a.end(), [&](auto u, auto v) {
    return fm(u) < fm(v);
  });
  return a;
}
