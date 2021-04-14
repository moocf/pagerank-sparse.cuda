#pragma once
#include <vector>

using std::vector;




template <class G, class K>
auto edge(G& x, K u) {
  for (K v : x.edges(u))
    return v;
  return K();
}


template <class G, class K>
auto edges(G& x, K u) {
  vector<K> a;
  for (auto v : x.edges(u))
    a.push_back(v);
  return a;
}

template <class G, class K>
auto inEdges(G& x, K v) {
  vector<K> a;
  for (auto u : x.inEdges(v))
    a.push_back(u);
  return a;
}




template <class G, class K>
bool edgesEqual(G& x, K u1, K u2) {
  auto i1 = x.edges(u1);
  auto i2 = x.edges(u2);
  auto b1 = i1.begin();
  auto b2 = i2.begin();
  auto e1 = i1.end();
  auto e2 = i2.end();
  for (; b1 != e1 && b2 != e2; b1++, b2++)
    if (*b1 != *b2) return false;
  return b1 == e1 && b2 == e2;
}

template <class G, class K>
bool inEdgesEqual(G& x, K u1, K u2) {
  auto i1 = x.edges(u1);
  auto i2 = x.edges(u2);
  auto b1 = i1.begin();
  auto b2 = i2.begin();
  auto e1 = i1.end();
  auto e2 = i2.end();
  for (; b1 != e1 && b2 != e2; b1++, b2++)
    if (*b1 != *b2) return false;
  return b1 == e1 && b2 == e2;
}
