#pragma once
#include <utility>
#include <cmath>
#include <string.h>
#include <omp.h>
#include "_cuda.h"
#include "fill.h"
#include "DiGraph.h"
#include "dotProduct.h"

using namespace std;




template <class T>
struct pageRankOptions {
  T damping;
  T convergence;

  pageRankOptions(T _damping=0.85, T _convergence=1e-6) {
    damping = _damping;
    convergence = _convergence;
  }
};




// Finds rank of nodes in graph.
template <class T>
void pageRank(vector<T>& a, DiGraph& x, T p, T E) {
  int S = x.span();
  int N = x.order();
  vector<T> r(S);
  fill(r, T(1)/N);
  while (1) {
    fill(a, (1-p)/N);
    for (int i=0; i<S; i++) {
      int d = x.degree(i);
      for (int j : x.edges(i))
        a[j] += p*r[i]/d;
    }
  }
}


template <class T>
void pageRank(T *a, T *w, int N, T p, T E) {
  T *r = new T[N], *a0 = a, *r0 = r;
  fill(r, N, T(1.0/N));
  while (1) {
    int e = 0;
    for (int j=0; j<N; j++) {
      T wjr = dotProduct(&w[N*j], r, N);
      a[j] = p*wjr + (1-p)/N;
      T ej = abs(r[j] - a[j]);
      if (ej >= E) e++;
    }
    swap(a, r);
    if (!e) break;
  }
  if (r != a0) memcpy(a0, r, N*sizeof(T));
  delete[] r0;
}


template <class T>
void pageRank(T *a, T *w, int N, pageRankOptions<T> o=pageRankOptions<T>()) {
  pageRank(a, w, N, o.damping, o.convergence);
}


template <class T>
void pageRank(T *a, DenseDiGraph<T>& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  pageRank(a, x.weights, x.order, o);
}




// Finds rank of nodes in graph.
template <class T>
void pageRankOmp(T *a, T *w, int N, T p, T E) {
  T *r = new T[N], *a0 = a, *r0 = r;
  fill(r, N, T(1.0/N));
  while (1) {
    int e = 0;
    #pragma omp parallel for
    for (int j=0; j<N; j++) {
      T wjr = dotProduct(&w[N*j], r, N);
      a[j] = p*wjr + (1-p)/N;
      T ej = abs(r[j] - a[j]);
      if (ej >= E) {
        #pragma omp atomic
        e++;
      }
    }
    swap(a, r);
    if (!e) break;
  }
  if (r != a0) memcpy(a0, r, N*sizeof(T));
  delete[] r0;
}


template <class T>
void pageRankOmp(T *a, T *w, int N, pageRankOptions<T> o=pageRankOptions<T>()) {
  pageRankOmp(a, w, N, o.damping, o.convergence);
}


template <class T>
void pageRankOmp(T *a, DenseDiGraph<T>& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  pageRankOmp(a, x.weights, x.order, o);
}




template <class T>
__global__ void pageRankKernel(int *e, T *a, T *r, T *w, int N, T p, T E) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  for (int j=b; j<N; j+=G) {
    cache[t] = dotProductKernelLoop(&w[N*j], r, N, t, B);
    sumKernelReduce(cache, B, t);
    if (t != 0) continue;
    T wjr = cache[0];
    a[j] = p*wjr + (1-p)/N;
    T ej = abs(a[j] - r[j]);
    if (ej >= E) atomicAdd(e, 1);
  }
}


template <class T>
void pageRankCuda(T *a, T *w, int N, T p, T E) {
  int threads = _THREADS;
  int blocks = max(ceilDiv(N, threads), 1024);
  size_t W1 = N*N * sizeof(T);
  size_t A1 = N * sizeof(T);
  size_t E1 = 1 * sizeof(int);
  int e = 0;

  int *eD;
  T *wD, *rD, *aD;
  TRY( cudaMalloc(&wD, W1) );
  TRY( cudaMalloc(&rD, A1) );
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&eD, E1) );
  TRY( cudaMemcpy(wD, w, W1, cudaMemcpyHostToDevice) );

  fillKernel<<<blocks, threads>>>(rD, N, T(1.0/N));
  while (1) {
    fillKernel<<<1, 1>>>(eD, 1, 0);
    pageRankKernel<<<blocks, threads>>>(eD, aD, rD, wD, N, p, E);
    TRY( cudaMemcpy(&e, eD, E1, cudaMemcpyDeviceToHost) );
    swap(aD, rD);
    if (!e) break;
  }
  TRY( cudaMemcpy(a, rD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(wD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(eD) );
}


template <class T>
void pageRankCuda(T *a, T *w, int N, pageRankOptions<T> o=pageRankOptions<T>()) {
  pageRankCuda(a, w, N, o.damping, o.convergence);
}


template <class T>
void pageRankCuda(T *a, DenseDiGraph<T>& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  pageRankCuda(a, x.weights, x.order, o);
}
