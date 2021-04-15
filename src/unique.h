#pragma once
#include <set>
#include <vector>

using std::set;
using std::vector;




template <class T>
bool isUnique(vector<vector<T>>& x) {
  set<T> s;
  for (auto& vs : x) {
    for (auto v : vs) {
      if (s.count(v)) return false;
      s.insert(s.end(), v);
    }
  }
  return true;
}
