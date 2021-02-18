#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include "_cuda.h"

using std::vector;
using std::unordered_map;
using std::max;




template <class T>
void add(T *x, int N, T v) {
  for (int i=0; i<N; i++)
    x[i] += v;
}

template <class T>
void add(vector<T>& x, T v) {
  add(x.data(), x.size(), v);
}

template <class K, class T>
void add(unordered_map<K, T>& x, T v) {
  for (auto& p : x) p.second += v;
}


template <class T, class C>
void addAt(T *x, C&& is , T v) {
  for (int i : is)
    x[i] += v;
}

template <class T, class C>
void addAt(vector<T>& x, C&& is, T v) {
  addAt(x.data(), is, v);
}

template <class K, class T, class C>
void addAt(unordered_map<K, T>& x, C&& ks, T v) {
  for (auto&& k : ks)
    x[k] += v;
}




template <class T>
void addOmp(T *x, int N, T v) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    x[i] += v;
}

template <class T>
void addOmp(vector<T>& x, T v) {
  addOmp(x.data(), x.size(), v);
}




template <class T>
__device__ void addKernelLoop(T *a, int N, T v, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] += v;
}


template <class T>
__global__ void addKernel(T *a, int N, T v) {
  DEFINE(t, b, B, G);

  addKernelLoop(a, N, v, B*b+t, G*B);
}


template <class T>
void addCuda(T *a, int N, T v) {
  int threads = _THREADS;
  int blocks = max(ceilDiv(N, threads), 1024);
  size_t A1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMemcpy(aD, a, A1, cudaMemcpyHostToDevice) );

  addKernel<<<blocks, threads>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}

template <class T>
void addCuda(vector<T>& x, T v) {
  addCuda(x.data(), x.size(), v);
}
