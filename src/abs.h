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




// ABS
// ---

template <class T>
void abs(T *a, int N) {
  for (int i=0; i<N; i++)
    a[i] = abs(a[i]);
}

template <class T>
void abs(vector<T>& a) {
  abs(a.begin(), a.end());
}

template <class K, class T>
void abs(unordered_map<K, T>& a) {
  for (auto& p : a)
    p.second = abs(p.second);
}




// ABS-AT
// ------

template <class T, class I>
void absAt(T *a, I&& is) {
  for (int i : is)
    a[i] = abs(a[i]);
}

template <class T, class I>
void absAt(vector<T>& a, I&& is) {
  absAt(a.data(), is);
}

template <class K, class T, class I>
void absAt(unordered_map<K, T>& a, I&& ks) {
  for (auto&& k : ks)
    a[k] = abs(a[k]);
}




// ABS (OMP)
// ---------

template <class T>
void absOmp(T *a, int N) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    a[i] = abs(a[i]);
}

template <class T>
void fillOmp(vector<T>& a) {
  absOmp(a.data(), a.size());
}




// ABS (CUDA)
// ----------

template <class T>
__device__ void absKernelLoop(T *a, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = abs(a[i]);
}


template <class T>
__global__ void fillKernel(T *a, int N) {
  DEFINE(t, b, B, G);

  absKernelLoop(a, N, B*b+t, G*B);
}


template <class T>
void absCuda(T *a, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMemcpy(aD, a, N1, cudaMemcpyHostToDevice) );

  absKernel<<<G, B>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}

template <class T>
void absCuda(vector<T>& a) {
  absCuda(a.data(), a.size());
}
