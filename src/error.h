#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include "_cuda.h"
#include "ceilDiv.h"
#include "sum.h"

using std::vector;
using std::unordered_map;
using std::abs;
using std::max;




// ABS-ERROR
// ---------

template <class T>
auto absError(T *x, T *y, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T>
auto absError(vector<T>& x, vector<T>& y) {
  return absError(x.data(), y.data(), x.size());
}

template <class K, class T>
auto absError(unordered_map<K, T>& x, unordered_map<K, T>& y) {
  T a = T();
  for (auto& p : x)
    a += abs(p.second - y[p.first]);
  return a;
}




// ABS-ERROR-ABS
// -------------

template <class T>
auto absErrorAbs(T *x, T *y, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += abs(abs(x[i]) - abs(y[i]));
  return a;
}

template <class T>
auto absErrorAbs(vector<T>& x, vector<T>& y) {
  return absErrorAbs(x.data(), y.data(), x.size());
}

template <class K, class T>
auto absErrorAbs(unordered_map<K, T>& x, unordered_map<K, T>& y) {
  T a = T();
  for (auto& p : x)
    a += abs(abs(p.second) - abs(y[p.first]));
  return a;
}




// ABS-ERROR (OMP)
// ---------------

template <class T>
auto absErrorOmp(T *x, T *y, int N) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T>
auto absErrorOmp(vector<T>& x, vector<T>& y) {
  return errorAbsOmp(x.data(), y.data(), x.size());
}




// ABS-ERROR (CUDA)
// ----------------

template <class T>
__device__ T absErrorKernelLoop(T *x, T *y, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += abs(x[i] - y[i]);
  return a;
}


template <class T>
__global__ void absErrorKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = absErrorKernelLoop(x, y, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
auto absErrorCuda(T *x, T *y, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);
  T a[GRID_DIM];

  T *xD, *yD, *aD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  absErrorKernel<<<G, B>>>(aD, xD, yD, N);
  TRY( cudaMemcpy(a, aD, G1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return sum(a, G);
}

template <class T>
auto absErrorCuda(vector<T>& x, vector<T>& y) {
  return absErrorCuda(x.data(), y.data(), x.size());
}




// ABS-ERROR-ABS (CUDA)
// --------------------

template <class T>
__device__ T absErrorAbsKernelLoop(T *x, T *y, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += abs(abs(x[i]) - abs(y[i]));
  return a;
}


template <class T>
__global__ void absErrorAbsKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = absErrorAbsKernelLoop(x, y, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
auto absErrorAbsCuda(T *x, T *y, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);
  T a[GRID_DIM];

  T *xD, *yD, *aD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  absErrorAbsKernel<<<G, B>>>(aD, xD, yD, N);
  TRY( cudaMemcpy(a, aD, G1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return sum(a, G);
}

template <class T>
auto absErrorAbsCuda(vector<T>& x, vector<T>& y) {
  return absErrorAbsCuda(x.data(), y.data(), x.size());
}
