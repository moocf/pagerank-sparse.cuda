#pragma once
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include "_cuda.h"
#include "ceilDiv.h"
#include "sum.h"

using namespace std;




// Finds absolute error between 2 vectors (arrays).
template <class T>
T errorAbs(T *x, T *y, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}


template <class T, size_t N>
T errorAbs(array<T, N>& x, array<T, N>& y) {
  return errorAbs(x.data(), y.data(), x.size());
}


template <class T>
T errorAbs(vector<T>& x, vector<T>& y) {
  return errorAbs(x.data(), y.data(), x.size());
}




template <class T>
T errorAbsOmp(T *x, T *y, int N) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}


template <class T, size_t N>
T errorAbsOmp(array<T, N>& x, array<T, N>& y) {
  return errorAbsOmp(x.data(), y.data(), x.size());
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
  __shared__ T cache[_THREADS];

  cache[t] = errorAbsKernelLoop(x, y, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
T errorAbsCuda(T *x, T *y, int N) {
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

  errorAbsKernel<<<blocks, threads>>>(aPartialD, xD, yD, N);
  TRY( cudaMemcpy(aPartial, aPartialD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aPartialD) );
  return sum(aPartial, blocks);
}


template <class T, size_t N>
T errorAbsCuda(array<T, N>& x, array<T, N>& y) {
  return errorAbsCuda(x.data(), y.data(), N);
}


template <class T>
T errorAbsCuda(vector<T>& x, vector<T>& y) {
  return errorAbsCuda(x.data(), y.data(), x.size());
}
