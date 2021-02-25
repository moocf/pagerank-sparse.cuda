#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <omp.h>
#include "_cuda.h"

using std::vector;
using std::unique_ptr;
using std::max;




template <class T>
T sum(T *x, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T>
T sum(vector<T>& x) {
  return sum(x.data(), x.size());
}

template <class K, class T>
T sum(unordered_map<K, T>& x) {
  T a = T();
  for (auto& p : x)
    a += p.second;
  return a;
}


template <class T, class C>
T sumAt(T *x, C&& is) {
  T a = T();
  for (int i : is)
    a += x[i];
  return a;
}

template <class T, class C>
T sumAt(vector<T>& x, C&& is) {
  return sumAt(x.data(), is);
}

template <class K, class T, class C>
T sumAt(unordered_map<K, T>& x, C&& ks) {
  T a = T();
  for (auto&& k : ks)
    a += x[k];
  return a;
}




template <class T>
T sumOmp(T *x, int N) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T>
T sumOmp(vector<T>& x) {
  return sumOmp(x.data(), x.size());
}




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
  __shared__ T cache[_THREADS];

  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
T sumCuda(T *x, int N) {
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), _BLOCKS);
  size_t X1 = N * sizeof(T);
  size_t A1 = blocks * sizeof(T);
  unique_ptr<T> a(new T[A1]);

  T *xD, *aD;
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );

  sumKernel<<<blocks, threads>>>(aD, xD, N);
  TRY( cudaMemcpy(a.get(), aD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return sum(a.get(), blocks);
}

template <class T>
T sumCuda(vector<T>& x) {
  return sumCuda(x.data(), x.size());
}




template <class T>
__device__ T sumAtKernelLoop(T *x, int *is, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[is[i]];
  return a;
}


template <class T>
__global__ void sumAtKernel(T *a, T *x, T *is, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  cache[t] = sumAtKernelLoop(x, is, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




template <class T, class C>
__device__ T sumIfNotKernelLoop(T *x, C *cs, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    if (cs[i] == 0) a += x[i];
  return a;
}


template <class T, class C>
__global__ void sumIfNotKernel(T *a, T *x, C *cs, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  cache[t] = sumIfNotKernelLoop(x, cs, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}
