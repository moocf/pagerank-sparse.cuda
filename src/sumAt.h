#pragma once
#include <array>
#include <vector>
#include <omp.h>
#include "_cuda.h"

// TODO
using namespace std;




template <class T>
T sumAt(T *x, int *is, int M) {
  T a = T();
  for (int m=0; m<M; m++)
    a += x[is[m]];
  return a;
}


template <class T, size_t N, size_t M>
T sumAt(array<T, N>& x, array<int, M>& is) {
  return sumAt(x.data(), is.data(), is.size());
}


template <class T>
T sumAt(vector<T>& x, vector<int>& is) {
  return sumAt(x.data(), is.data(), is.size());
}




template <class T>
T sumAtOmp(T *x, int *is, int M) {
  T a = T();
  #pragma omp parallel for reduction (+:a)
  for (int m=0; m<M; m++)
    a += x[is[m]];
  return a;
}


template <class T, size_t N, size_t M>
T sumAtOmp(array<T, N>& x, array<int, M>& is) {
  return sumAtOmp(x.data(), is.data(), is.size());
}


template <class T>
T sumAtOmp(vector<T>& x, vector<int>& is) {
  return sumAtOmp(x.data(), is.data(), is.size());
}




template <class T>
__device__ T sumAtKernelLoop(T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


template <class T>
__global__ void sumAtKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
T sumAtCuda(T *x, int N) {
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
T sumAtCuda(array<T, N>& x) {
  return sumCuda(x.data(), x.size());
}


template <class T>
T sumAtCuda(vector<T>& x) {
  return sumCuda(x.data(), x.size());
}
