#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include "_cuda.h"

using std::vector;
using std::unordered_map;
using std::max;




// FILL
// ----

template <class T>
void fill(T *a, int N, T v) {
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T>
void fill(vector<T>& a, T v) {
  fill(a.begin(), a.end(), v);
}

template <class K, class T>
void fill(unordered_map<K, T>& a, T v) {
  for (auto& p : a)
    p.second = v;
}




// FILL-AT
// -------

template <class T, class I>
void fillAt(T *a, T v, I&& is) {
  for (int i : is)
    a[i] = v;
}

template <class T, class I>
void fillAt(vector<T>& a, T v, I&& is) {
  fillAt(a.data(), v, is);
}

template <class K, class T, class I>
void fillAt(unordered_map<K, T>& a, T v, I&& ks) {
  for (auto&& k : ks)
    a[k] = v;
}




// FILL (OMP)
// ----------

template <class T>
void fillOmp(T *a, int N, T v) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T>
void fillOmp(vector<T>& a, T v) {
  fillOmp(a.data(), a.size(), v);
}




// FILL (CUDA)
// -----------

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
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMemcpy(aD, a, N1, cudaMemcpyHostToDevice) );

  fillKernel<<<G, B>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}

template <class T>
void fillCuda(vector<T>& a, T v) {
  fillCuda(a.data(), a.size(), v);
}




// FILL-AT (CUDA)
// --------------

template <class T>
__device__ void fillAtKernelLoop(T *a, T v, int *is, int IS, int i, int DI) {
  for (; i<IS; i+=DI)
    a[is[i]] = v;
}


template <class T>
__global__ void fillAtKernel(T *a, T v, int *is, int IS) {
  DEFINE(t, b, B, G);

  fillAtKernelLoop(a, v, is, IS, B*b+t, G*B);
}
