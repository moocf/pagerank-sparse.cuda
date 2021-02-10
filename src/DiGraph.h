#pragma once
#include <vector>
#include <algorithm>

using namespace std;




class DiGraph {
  vector<bool> has;
  vector<vector<int>> out;
  int N = 0, M = 0;




  public:
  int order() {
    return N;
  }


  int size() {
    return M;
  }


  int span() {
    return has.size();
  }




  vector<int> vertices() {
    vector<int> a;
    a.reserve(N);
    for (int i=0, I=has.size(); i<I; i++)
      if (has[i]) a.push_back(i);
    return a;
  }


  bool hasVertex(int i) {
    if (i >= span()) return false;
    return has[i];
  }


  void addVertex(int i) {
    if (hasVertex(i)) return;
    if (i >= span()) {
      has.resize(i+1);
      out.resize(i+1);
    }
    has[i] = true;
    N++;
  }


  void removeVertex(int i) {
    if (!hasVertex(i)) return;
    out[i].clear();
    has[i] = false;
    N--;
  }




  vector<int>& edges(int i) {
    return out[i];
  }


  int degree(int i) {
    return edges(i).size();
  }


  bool hasEdge(int i, int j) {
    if (i >= span() || j >= span()) return false;
    auto& e = edges(i);
    return find(e.begin(), e.end(), j) != e.end();
  }


  void addEdge(int i, int j) {
    if (hasEdge(i, j)) return;
    addVertex(i);
    addVertex(j);
    out[i].push_back(j);
    M++;
  }


  void removeEdge(int i, int j) {
    if (!hasEdge(i, j)) return;
    auto& e = edges(i);
    e.erase(remove(e.begin(), e.end(), j), e.end());
    M--;
  }
};
