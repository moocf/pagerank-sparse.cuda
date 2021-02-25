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
void fill(T *x, int N, T v) {
  for (int i=0; i<N; i++)
    x[i] = v;
}

template <class T>
void fill(vector<T>& x, T v) {
  fill(x.data(), x.size(), v);
}

template <class K, class T>
void fill(unordered_map<K, T>& x, T v) {
  for (auto& p : x) p.second = v;
}


template <class T, class C>
void fillAt(T *x, C&& is , T v) {
  for (int i : is)
    x[i] = v;
}

template <class T, class C>
void fillAt(vector<T>& x, C&& is, T v) {
  fillAt(x.data(), is, v);
}

template <class K, class T, class C>
void fillAt(unordered_map<K, T>& x, C&& ks, T v) {
  for (auto&& k : ks)
    x[k] = v;
}




template <class T>
void fillOmp(T *x, int N, T v) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    x[i] = v;
}

template <class T>
void fillOmp(vector<T>& x, T v) {
  fillOmp(x.data(), x.size(), v);
}




template <class T>
__device__ void fillKernelLoop(T *a, int N, T v, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = v;
}


template <class T>
__global__ void fillKernel(T *a, int N, T v) {
  DEFINE(t, b, B, G);

  fillKernelLoop(a, N, v, B*b+t, G*B);
}


template <class T>
void fillCuda(T *a, int N, T v) {
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), _BLOCKS);
  size_t A1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMemcpy(aD, a, A1, cudaMemcpyHostToDevice) );

  fillKernel<<<blocks, threads>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}

template <class T>
void fillCuda(vector<T>& x, T v) {
  fillCuda(x.data(), x.size(), v);
}
