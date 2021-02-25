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
void multiply(T *x, int N, T v) {
  for (int i=0; i<N; i++)
    x[i] *= v;
}

template <class T>
void multiply(vector<T>& x, T v) {
  multiply(x.data(), x.size(), v);
}

template <class K, class T>
void multiply(unordered_map<K, T>& x, T v) {
  for (auto& p : x) p.second *= v;
}




template <class T>
void multiply(T *a, T *x, T *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T>
void multiply(vector<T>& a, vector<T>& x, vector<T>& y) {
  multiply(a.data(), x.data(), y.data(), a.size());
}

template <class K, class T>
void multiply(unordered_map<K, T>& a, unordered_map<K, T>& x, unordered_map<K, T>& y) {
  for (auto& p : x)
    a[p.first] = x[p.first] * y[p.first];
}




template <class T, class C>
void multiplyAt(T *x, C&& is , T v) {
  for (int i : is)
    x[i] *= v;
}

template <class T, class C>
void multiplyAt(vector<T>& x, C&& is, T v) {
  multiplyAt(x.data(), is, v);
}

template <class K, class T, class C>
void multiplyAt(unordered_map<K, T>& x, C&& ks, T v) {
  for (auto&& k : ks)
    x[k] *= v;
}




template <class T>
void multiplyOmp(T *x, int N, T v) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    x[i] *= v;
}

template <class T>
void multiplyOmp(vector<T>& x, T v) {
  multiplyOmp(x.data(), x.size(), v);
}




template <class T>
__device__ void multiplyKernelLoop(T *a, int N, T v, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] *= v;
}


template <class T>
__global__ void multiplyKernel(T *a, int N, T v) {
  DEFINE(t, b, B, G);

  multiplyKernelLoop(a, N, v, B*b+t, G*B);
}


template <class T>
void multiplyCuda(T *a, int N, T v) {
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), _BLOCKS);
  size_t A1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMemcpy(aD, a, A1, cudaMemcpyHostToDevice) );

  multiplyKernel<<<blocks, threads>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}

template <class T>
void multiplyCuda(vector<T>& x, T v) {
  multiplyCuda(x.data(), x.size(), v);
}




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
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), _BLOCKS);
  size_t A1 = N * sizeof(T);

  T *xD, *yD;
  TRY( cudaMalloc(&xD, A1) );
  TRY( cudaMalloc(&yD, A1) );
  TRY( cudaMemcpy(xD, x, A1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, A1, cudaMemcpyHostToDevice) );

  multiplyKernel<<<blocks, threads>>>(xD, xD, yD, N);
  TRY( cudaMemcpy(a, xD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(yD) );
}

template <class T>
void multiplyCuda(vector<T>& a, vector<T>& x, vector<T> &y) {
  multiplyCuda(a.data(), x.data(), y.data(), a.size());
}
