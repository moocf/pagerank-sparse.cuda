#pragma once
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include "_cuda.h"
#include "ceilDiv.h"
#include "sum.h"

using std::vector;
using std::unique_ptr;
using std::abs;
using std::max;




// Finds absolute error between 2 vectors.
template <class T>
T errorAbs(T *x, T *y, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T>
T errorAbs(vector<T>& x, vector<T>& y) {
  return errorAbs(x.data(), y.data(), x.size());
}

template <class K, class T>
T errorAbs(unordered_map<K, T>& x, unordered_map<K, T>& y) {
  T a = T();
  for (auto& p : x)
    a += abs(p.second - y[p.first]);
  return a;
}




template <class T>
T errorAbsOmp(T *x, T *y, int N) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T>
T errorAbsOmp(vector<T>& x, vector<T>& y) {
  return errorAbsOmp(x.data(), y.data(), x.size());
}




template <class T>
__device__ T errorAbsKernelLoop(T *x, T *y, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += abs(x[i] - y[i]);
  return a;
}


template <class T>
__global__ void errorAbsKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = errorAbsKernelLoop(x, y, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
T errorAbsCuda(T *x, T *y, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);
  unique_ptr<T> a(new T[G1]);

  T *xD, *yD, *aD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  errorAbsKernel<<<G, B>>>(aD, xD, yD, N);
  TRY( cudaMemcpy(a.get(), aD, G1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return sum(a.get(), G);
}

template <class T>
T errorAbsCuda(vector<T>& x, vector<T>& y) {
  return errorAbsCuda(x.data(), y.data(), x.size());
}
