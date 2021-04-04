#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "_cuda.h"

using std::vector;
using std::unordered_map;
using std::max;
using std::abs;




// SUM
// ---

template <class T>
auto sum(T *x, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T>
auto sum(vector<T>& x) {
  return sum(x.data(), x.size());
}

template <class K, class T>
auto sum(unordered_map<K, T>& x) {
  T a = T();
  for (auto&& p : x)
    a += p.second;
  return a;
}




// SUM-ABS
// -------

template <class T>
auto sumAbs(T *x, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += abs(x[i]);
  return a;
}

template <class T>
auto sumAbs(vector<T>& x) {
  return sumAbs(x.data(), x.size());
}

template <class K, class T>
auto sumAbs(unordered_map<K, T>& x) {
  T a = T();
  for (auto&& p : x)
    a += abs(p.second);
  return a;
}




// SUM-AT
// ------

template <class T, class I>
auto sumAt(T *x, I&& is) {
  T a = T();
  for (int i : is)
    a += x[i];
  return a;
}

template <class T, class I>
auto sumAt(vector<T>& x, I&& is) {
  return sumAt(x.data(), is);
}

template <class K, class T, class I>
auto sumAt(unordered_map<K, T>& x, I&& ks) {
  T a = T();
  for (auto&& k : ks)
    a += x[k];
  return a;
}




// SUM-ABS-AT
// ----------

template <class T, class I>
auto sumAbsAt(T *x, I&& is) {
  T a = T();
  for (int i : is)
    a += abs(x[i]);
  return a;
}

template <class T, class I>
auto sumAbsAt(vector<T>& x, I&& is) {
  return sumAbsAt(x.data(), is);
}

template <class K, class T, class I>
auto sumAbsAt(unordered_map<K, T>& x, I&& ks) {
  T a = T();
  for (auto&& k : ks)
    a += abs(x[k]);
  return a;
}




// SUM (OMP)
// ---------

template <class T>
auto sumOmp(T *x, int N) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T>
auto sumOmp(vector<T>& x) {
  return sumOmp(x.data(), x.size());
}




// SUM (CUDA)
// ----------

template <class T>
__device__ void sumKernelReduce(T* a, int N, int i) {
  __syncthreads();
  for (N=N/2; N>0; N/=2) {
    if (i < N) a[i] += a[N+i];
    __syncthreads();
  }
}


template <class T>
__device__ T sumKernelLoop(T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


template <class T>
__global__ void sumKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
auto sumCuda(T *x, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);
  T a[GRID_DIM];

  T *xD, *aD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );

  sumKernel<<<G, B>>>(aD, xD, N);
  TRY( cudaMemcpy(a, aD, G1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return sum(a, G);
}

template <class T>
auto sumCuda(vector<T>& x) {
  return sumCuda(x.data(), x.size());
}




// SUM-ABS (CUDA)
// --------------

template <class T>
__device__ T sumAbsKernelLoop(T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += abs(x[i]);
  return a;
}


template <class T>
__global__ void sumAbsKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumAbsKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
auto sumAbsCuda(T *x, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);
  size_t G1 = G * sizeof(T);
  T a[GRID_DIM];

  T *xD, *aD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&aD, G1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );

  sumAbsKernel<<<G, B>>>(aD, xD, N);
  TRY( cudaMemcpy(a, aD, G1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return sum(a, G);
}

template <class T>
auto sumAbsCuda(vector<T>& x) {
  return sumAbsCuda(x.data(), x.size());
}




// SUM-AT (CUDA)
// -------------

template <class T>
__device__ T sumAtKernelLoop(T *x, int *is, int IS, int i, int DI) {
  T a = T();
  for (; i<IS; i+=DI)
    a += x[is[i]];
  return a;
}


template <class T>
__global__ void sumAtKernel(T *a, T *x, T *is, int IS) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumAtKernelLoop(x, is, IS, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




// SUM-ABS-AT (CUDA)
// -----------------

template <class T>
__device__ T sumAbsAtKernelLoop(T *x, int *is, int IS, int i, int DI) {
  T a = T();
  for (; i<IS; i+=DI)
    a += abs(x[is[i]]);
  return a;
}


template <class T>
__global__ void sumAbsAtKernel(T *a, T *x, T *is, int IS) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumAbsAtKernelLoop(x, is, IS, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




// SUM-IF-NOT (CUDA)
// -----------------

template <class T, class C>
__device__ T sumIfNotKernelLoop(T *x, C *cs, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    if (!cs[i]) a += x[i];
  return a;
}


template <class T, class C>
__global__ void sumIfNotKernel(T *a, T *x, C *cs, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumIfNotKernelLoop(x, cs, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




// SUM-ABS-IF-NOT (CUDA)
// ---------------------

template <class T, class C>
__device__ T sumAbsIfNotKernelLoop(T *x, C *cs, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    if (cs[i] == 0) a += abs(x[i]);
  return a;
}


template <class T, class C>
__global__ void sumAbsIfNotKernel(T *a, T *x, C *cs, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumAbsIfNotKernelLoop(x, cs, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}
