#pragma once
#include <vector>

using std::vector;




// Traverses nodes in depth-first manner, listing on entry.
template <class G, class C, class K>
void dfsLoop(G& x, K u, C& vis, vector<K>& a) {
  vis[u] = true;
  a.push_back(u);
  for (auto& v : x.edges(u))
    if (!vis[v]) dfsLoop(x, v, vis, a);
}

template <class G, class K>
auto dfs(G& x, K u) {
  auto vis = x.vertexContainer(bool());
  vector<K> a;
  dfsLoop(x, u, vis, a);
  return a;
}




// Traverses nodes in depth-first manner, listing on exit.
template <class G, class C, class K>
void dfsEndLoop(G& x, K u, C& vis, vector<K>& a) {
  vis[u] = true;
  for (auto& v : x.edges(u))
    if (!vis[v]) dfsEndLoop(x, v, vis, a);
  a.push_back(u);
}

template <class G, class K>
auto dfsEnd(G& x, K u) {
  auto vis = x.vertexContainer(bool());
  vector<K> a;
  dfsEndLoop(x, u, vis, a);
  return a;
}
