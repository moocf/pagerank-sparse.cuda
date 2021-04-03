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




// ADD-VALUE
// ---------

template <class T>
void addValue(T *a, int N, T v) {
  for (int i=0; i<N; i++)
    a[i] += v;
}

template <class T>
void addValue(vector<T>& a, T v) {
  addValue(a.data(), a.size(), v);
}

template <class K, class T>
void addValue(unordered_map<K, T>& a, T v) {
  for (auto&& p : a)
    p.second += v;
}




// ADD-VALUE-AT
// ------------

template <class T, class I>
void addValueAt(T *a, I&& is , T v) {
  for (int i : is)
    a[i] += v;
}

template <class T, class I>
void addValueAt(vector<T>& a, I&& is, T v) {
  addValueAt(a.data(), is, v);
}

template <class K, class T, class I>
void addValueAt(unordered_map<K, T>& a, I&& ks, T v) {
  for (auto&& k : ks)
    a[k] += v;
}




// ADD
// ---

template <class T>
void add(T *a, T *x, T *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i] + y[i];
}

template <class T>
void add(vector<T>& a, vector<T>& x, vector<T>& y) {
  return add(a.data(), x.data(), y.data(), x.size());
}

template <class K, class T>
void add(unordered_map<K, T>& a, unordered_map<K, T>& x, unordered_map<K, T>& y) {
  for (auto&& p : x)
    a[p.first] = x[p.first] + y[p.first];
}




// ADD-ABS
// -------

template <class T>
void addAbs(T *a, T *x, T *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = abs(x[i] + y[i]);
}

template <class T>
void addAbs(vector<T>& a, vector<T>& x, vector<T>& y) {
  return addAbs(a.data(), x.data(), y.data(), x.size());
}

template <class K, class T>
void addAbs(unordered_map<K, T>& a, unordered_map<K, T>& x, unordered_map<K, T>& y) {
  for (auto&& p : x)
    a[p.first] = abs(x[p.first] + y[p.first]);
}




// ADD-VALUE (OMP)
// ---------------

template <class T>
void addValueOmp(T *a, int N, T v) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    a[i] += v;
}

template <class T>
void addValueOmp(vector<T>& a, T v) {
  addValueOmp(a.data(), a.size(), v);
}




// ADD-VALUE (CUDA)
// ----------------

template <class T>
__device__ void addValueKernelLoop(T *a, int N, T v, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] += v;
}


template <class T>
__global__ void addValueKernel(T *a, int N, T v) {
  DEFINE(t, b, B, G);

  addValueKernelLoop(a, N, v, B*b+t, G*B);
}


template <class T>
void addValueCuda(T *a, int N, T v) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMemcpy(aD, a, N1, cudaMemcpyHostToDevice) );

  addValueKernel<<<G, B>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}

template <class T>
void addValueCuda(vector<T>& a, T v) {
  addValueCuda(a.data(), a.size(), v);
}




// ADD (CUDA)
// ----------

template <class T>
__device__ void addKernelLoop(T *a, T *x, T *y, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = x[i] + y[i];
}


template <class T>
__global__ void addKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);

  addKernelLoop(a, x, y, N, B*b+t, G*B);
}


template <class T>
void addCuda(T *a, T *x, T *y, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *xD, *yD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  addKernel<<<G, B>>>(xD, xD, yD, N);
  TRY( cudaMemcpy(a, xD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(yD) );
}

template <class T>
void addCuda(vector<T>& a, vector<T>& x, vector<T>& y) {
  addCuda(a.data(), x.data(), y.data(), x.size());
}




// ADD-ABS (CUDA)
// --------------

template <class T>
__device__ void addAbsKernelLoop(T *a, T *x, T *y, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = abs(x[i] + y[i]);
}


template <class T>
__global__ void addAbsKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);

  addAbsKernelLoop(a, x, y, N, B*b+t, G*B);
}


template <class T>
void addAbsCuda(T *a, T *x, T *y, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *xD, *yD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  addAbsKernel<<<G, B>>>(xD, xD, yD, N);
  TRY( cudaMemcpy(a, xD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(yD) );
}

template <class T>
void addAbsCuda(vector<T>& a, vector<T>& x, vector<T>& y) {
  addAbsCuda(a.data(), x.data(), y.data(), x.size());
}
