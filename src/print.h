#pragma once
#include <array>
#include <vector>
#include <stdio.h>
#include "_support.h"




// Prints 1D array.
template <class T>
void print(T *x, int N) {
  printf("{");
  for (int i=0; i<N-1; i++)
    printf("%.4f, ", x[i]);
  if (N>0) printf("%.4f", x[N-1]);
  printf("}\n");
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
