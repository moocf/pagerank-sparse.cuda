#pragma once
#include "_cuda.h"




template <class T>
bool getBit(T *x, int i) {
  int W = 8 * sizeof(T);
  return x[i/W] >> (i%W);
}


template <class T>
__device__ bool getBit(T *x, int i) {
  int W = 8 * sizeof(T);
  return x[i/W] >> (i%W);
}




template <class T>
void setBit(T *x, int i) {
  int W = 8 * sizeof(T);
  x[i/W] |= 1 << (i%W);
}


template <class T>
__device__ void setBit(T *x, int i) {
  int W = 8 * sizeof(T);
  x[i/W] |= 1 << (i%W);
}
