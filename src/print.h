#pragma once
#include <vector>
#include <unordered_map>
#include <iostream>
#include "_support.h"
#include "DiGraph.h"

using std::vector;
using std::unordered_map;
using std::cout;



// Prints 1D array.
template <class T>
void print(T *x, int N) {
  cout << "{";
  for (int i=0; i<N; i++)
    cout << " " << x[i];
  cout << " }\n";
}

template <class T>
void print(vector<T>& x) {
  print(x.data(), x.size());
}

template <class K, class T>
void print(unordered_map<K, T>& x) {
  cout << "{\n";
  for (auto& p : x)
    cout << "  " << p.first << " => " << p.second << "\n";
  cout << "}\n";
}




// Prints 2D array.
template <class T>
void print(T *x, int R, int C) {
  cout << "{\n";
  for (int r=0; r<R; r++) {
    for (int c=0; c<C; c++)
      cout << "  " << GET2D(x, r, c, C);
    cout << "\n";
  }
  cout << "}\n";
}




// Prints graph.
template <class G>
void print(G& x, bool all=false) {
  cout << "order: " << x.order() << " size: " << x.size();
  if (!all) { cout << " {}\n"; return; }
  cout << " {\n";
  for (auto u : x.vertices()) {
    cout << "  " << u << " ->";
    for (auto v : x.edges(u))
      cout << " " << v;
    cout << "\n";
  }
  cout << "}\n";
}
