#pragma once
#include <array>
#include <vector>
#include <omp.h>
#include "_cuda.h"

using namespace std;




template <class T>
T sum(T *x, int N) {
  T a = T();
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}


template <class T, size_t N>
T sum(array<T, N>& x) {
  return sum(x.data(), x.size());
}


template <class T>
T sum(vector<T>& x) {
  return sum(x.data(), x.size());
}




template <class T>
T sumOmp(T *x, int N) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}


template <class T, size_t N>
T sumOmp(array<T, N>& x) {
  return sumOmp(x.data(), x.size());
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
  int blocks = max(ceilDiv(N, threads), 1024);
  size_t X1 = N * sizeof(T);
  size_t A1 = blocks * sizeof(T);
  T *aPartial = (T*) malloc(A1);

  T *xD, *aPartialD;
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&aPartialD, A1) );
  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );

  sumKernel<<<blocks, threads>>>(aPartialD, xD, N);
  TRY( cudaMemcpy(aPartial, aPartialD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(aPartialD) );
  return sum(aPartial, blocks);
}


template <class T, size_t N>
T sumCuda(array<T, N>& x) {
  return sumCuda(x.data(), x.size());
}


template <class T>
T sumCuda(vector<T>& x) {
  return sumCuda(x.data(), x.size());
}
