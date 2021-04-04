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




// MULTIPLY-VALUE
// --------------

template <class T>
void multiplyValue(T *a, int N, T v) {
  for (int i=0; i<N; i++)
    a[i] *= v;
}

template <class T>
void multiplyValue(vector<T>& a, T v) {
  multiplyValue(a.data(), a.size(), v);
}

template <class K, class T>
void multiplyValue(unordered_map<K, T>& a, T v) {
  for (auto& p : a)
    p.second *= v;
}




// MULTIPLY-VALUE-AT
// -----------------

template <class T, class I>
void multiplyValueAt(T *a, T v, I&& is) {
  for (int i : is)
    a[i] *= v;
}

template <class T, class I>
void multiplyValueAt(vector<T>& a, T v, I&& is) {
  multiplyValueAt(a.data(), v, is);
}

template <class K, class T, class I>
void multiplyValueAt(unordered_map<K, T>& a, T v, I&& ks) {
  for (auto&& k : ks)
    a[k] *= v;
}




// MULTIPLY
// --------

template <class T>
void multiply(T *a, T *x, T *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T>
void multiply(vector<T>& a, vector<T>& x, vector<T>& y) {
  multiply(a.data(), x.data(), y.data(), x.size());
}

template <class K, class T>
void multiply(unordered_map<K, T>& a, unordered_map<K, T>& x, unordered_map<K, T>& y) {
  for (auto&& p : x)
    a[p.first] = x[p.first] * y[p.first];
}




// MULTIPLY-ABS
// ------------

template <class T>
void multiplyAbs(T *a, T *x, T *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = abs(x[i] * y[i]);
}

template <class T>
void multiplyAbs(vector<T>& a, vector<T>& x, vector<T>& y) {
  multiplyAbs(a.data(), x.data(), y.data(), x.size());
}

template <class K, class T>
void multiplyAbs(unordered_map<K, T>& a, unordered_map<K, T>& x, unordered_map<K, T>& y) {
  for (auto&& p : x)
    a[p.first] = abs(x[p.first] * y[p.first]);
}




// MULTIPLY-VALUE (OMP)
// --------------------

template <class T>
void multiplyValueOmp(T *a, int N, T v) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    a[i] *= v;
}

template <class T>
void multiplyValueOmp(vector<T>& a, T v) {
  multiplyValueOmp(a.data(), a.size(), v);
}




// MULTIPLY-VALUE (CUDA)
// ---------------------

template <class T>
__device__ void multiplyValueKernelLoop(T *a, int N, T v, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] *= v;
}


template <class T>
__global__ void multiplyValueKernel(T *a, int N, T v) {
  DEFINE(t, b, B, G);

  multiplyValueKernelLoop(a, N, v, B*b+t, G*B);
}


template <class T>
void multiplyValueCuda(T *a, int N, T v) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMemcpy(aD, a, N1, cudaMemcpyHostToDevice) );

  multiplyValueKernel<<<G, B>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}

template <class T>
void multiplyValueCuda(vector<T>& a, T v) {
  multiplyValueCuda(a.data(), a.size(), v);
}




// MULTIPLY (CUDA)
// ---------------

template <class T>
__device__ void multiplyKernelLoop(T *a, T *x, T *y, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = x[i] * y[i];
}


template <class T>
__global__ void multiplyKernel(T *a, T *x, T* y, int N) {
  DEFINE(t, b, B, G);

  multiplyKernelLoop(a, x, y, N, B*b+t, G*B);
}


template <class T>
void multiplyCuda(T *a, T *x, T *y, int N, T v) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *xD, *yD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  multiplyKernel<<<G, B>>>(xD, xD, yD, N);
  TRY( cudaMemcpy(a, xD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(yD) );
}

template <class T>
void multiplyCuda(vector<T>& a, vector<T>& x, vector<T>& y) {
  multiplyCuda(a.data(), x.data(), y.data(), x.size());
}




// MULTIPLY-ABS (CUDA)
// -------------------

template <class T>
__device__ void multiplyAbsKernelLoop(T *a, T *x, T *y, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = abs(x[i] * y[i]);
}


template <class T>
__global__ void multiplyAbsKernel(T *a, T *x, T* y, int N) {
  DEFINE(t, b, B, G);

  multiplyAbsKernelLoop(a, x, y, N, B*b+t, G*B);
}


template <class T>
void multiplyAbsCuda(T *a, T *x, T *y, int N, T v) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *xD, *yD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  multiplyAbsKernel<<<G, B>>>(xD, xD, yD, N);
  TRY( cudaMemcpy(a, xD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(yD) );
}

template <class T>
void multiplyAbsCuda(vector<T>& a, vector<T>& x, vector<T>& y) {
  multiplyAbsCuda(a.data(), x.data(), y.data(), x.size());
}
