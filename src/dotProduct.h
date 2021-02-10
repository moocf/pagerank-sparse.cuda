#pragma once
#include <array>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "_cuda.h"
#include "ceilDiv.h"
#include "sum.h"

using namespace std;




// Finds sum of element-by-element product of 2 vectors (arrays).
template <class T>
T dotProduct(T *x, T *y, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i] * y[i];
  return a;
}


template <class T, size_t N>
T dotProduct(array<T, N>& x, array<T, N>& y) {
  return dotProduct(x.data(), y.data(), x.size());
}


template <class T>
T dotProduct(vector<T>& x, vector<T>& y) {
  return dotProduct(x.data(), y.data(), x.size());
}




template <class T>
T dotProductOmp(T *x, T *y, int N) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int i=0; i<N; i++)
    a += x[i] * y[i];
  return a;
}


template <class T, size_t N>
T dotProductOmp(array<T, N>& x, array<T, N>& y) {
  return dotProductOmp(x.data(), y.data(), x.size());
}


template <class T>
T dotProductOmp(vector<T>& x, vector<T>& y) {
  return dotProductOmp(x.data(), y.data(), x.size());
}




template <class T>
__device__ T dotProductKernelLoop(T *x, T *y, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i] * y[i];
  return a;
}


template <class T>
__global__ void dotProductKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  cache[t] = dotProductKernelLoop(x, y, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
T dotProductCuda(T *x, T *y, int N) {
  int threads = _THREADS;
  int blocks = max(ceilDiv(N, threads), 1024);
  size_t X1 = N * sizeof(T);
  size_t A1 = blocks * sizeof(T);
  T *aPartial = (T*) malloc(A1);

  T *xD, *yD, *aPartialD;
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&yD, X1) );
  TRY( cudaMalloc(&aPartialD, A1) );
  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, X1, cudaMemcpyHostToDevice) );

  dotProductKernel<<<blocks, threads>>>(aPartialD, xD, yD, N);
  TRY( cudaMemcpy(aPartial, aPartialD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aPartialD) );
  return sum(aPartial, blocks);
}


template <class T, size_t N>
T dotProductCuda(array<T, N>& x, array<T, N>& y) {
  return dotProductCuda(x.data(), y.data(), N);
}


template <class T>
T dotProductCuda(vector<T>& x, vector<T>& y) {
  return dotProductCuda(x.data(), y.data(), x.size());
}
