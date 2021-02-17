#pragma once
#include <array>
#include <vector>
#include <iostream>
#include "_support.h"
#include "DiGraphTemp.h"

using std::array;
using std::vector;
using std::cout;



// Prints 1D array.
template <class T>
void print(T *x, int N) {
  cout << "{";
  for (int i=0; i<N-1; i++)
    cout << x[i] << ", ";
  if (N>0) cout << x[N-1];
  cout << "}\n";
}


template <class T, size_t N>
void print(array<T, N>& x) {
  print(x.data(), x.size());
}


template <class T>
void print(vector<T>& x) {
  print(x.data(), x.size());
}




// Prints 2D array.
template <class T>
void print(T *x, int R, int C) {
  printf("{\n");
  for (int r=0; r<R; r++) {
    for (int c=0; c<C; c++)
      printf("%.4f, ", GET2D(x, r, c, C));
    printf("\n");
  }
  printf("}\n");
}




// Prints graph.
template <class K, class V, class E>
void print(DiGraphTemp<K, V, E>& x, bool all=false) {
  printf("order: %d, size: %d", x.order(), x.size());
  if (!all) { printf("\n"); return; }
  printf("{\n");
  for (int i=0, I=x.order(); i<I; i++) {
    if (!x.hasVertex(i)) continue;
    printf("%d ->", i);
    for (int j : x.edges(i))
      printf(" %d", j);
    printf("\n");
  }
  printf("}\n");
}
