#pragma once
#include <vector>

using namespace std;




class DiGraph {
  vector<bool> has;
  vector<vector<int>> out;
  int N, M;



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


  bool hasVertex(int i) {
    return has[i];
  }

  vector<int> vertices() {
    vector<int> a;
    a.reserve(N);
    for (int i=0, I=has.size(); i<I; i++)
      if (has[i]) a.push_back(i);
    return a;
  }


  int degree(int i) {
    return out.size();
  }

  vector<int>& edges(int i) {
    return out[i];
  }
};
