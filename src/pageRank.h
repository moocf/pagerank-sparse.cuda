#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <omp.h>
#include "_cuda.h"
#include "fill.h"
#include "DiGraph.h"
#include "errorAbs.h"

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
void pageRankPush(DiGraph& x, T p, T r, vector<T>& a) {
  int S = x.span(), N = x.order();
  for (int j=0; j<S; j++)
    if (x.hasVertex(j)) a[j] += p*r/N;
}


template <class T>
void pageRankPush(vector<int>& es, T p, T r, vector<T>& a) {
  int d = es.size();
  for (int j : es)
    a[j] += p*r/d;
}


template <class T>
void pageRankStep(DiGraph& x, T p, vector<T>& r, vector<T>& a) {
  int S = x.span(), N = x.order();
  fill(a, (1-p)/N);
  for (int i=0; i<S; i++) {
    if (!x.hasVertex(i)) continue;
    int d = x.degree(i);
    if (d == 0) pageRankPush(x, p, r[i], a);
    else pageRankPush(x.edges(i), p, r[i], a);
  }
}


template <class T>
vector<T>& pageRank(DiGraph& x, T p, T E) {
  int S = x.span(), N = x.order();
  vector<T>& r = *new vector<T>(S);
  vector<T>& a = *new vector<T>(S);
  fill(r, T(1)/N);
  while (1) {
    pageRankStep(x, p, r, a);
    T e = errorAbs(a, r);
    if (e < E) break;
    swap(a, r);
  }
  return r;
}


template <class T=float>
vector<T>& pageRank(DiGraph& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRank(x, o.damping, o.convergence);
}




// template <class T>
// __global__ void pageRankKernel(int *e, T *a, T *r, T *w, int N, T p, T E) {
//   DEFINE(t, b, B, G);
//   __shared__ T cache[_THREADS];

//   for (int j=b; j<N; j+=G) {
//     cache[t] = dotProductKernelLoop(&w[N*j], r, N, t, B);
//     sumKernelReduce(cache, B, t);
//     if (t != 0) continue;
//     T wjr = cache[0];
//     a[j] = p*wjr + (1-p)/N;
//     T ej = abs(a[j] - r[j]);
//     if (ej >= E) atomicAdd(e, 1);
//   }
// }


// template <class T>
// void pageRankCuda(T *a, T *w, int N, T p, T E) {
//   int threads = _THREADS;
//   int blocks = max(ceilDiv(N, threads), 1024);
//   size_t W1 = N*N * sizeof(T);
//   size_t A1 = N * sizeof(T);
//   size_t E1 = 1 * sizeof(int);
//   int e = 0;

//   int *eD;
//   T *wD, *rD, *aD;
//   TRY( cudaMalloc(&wD, W1) );
//   TRY( cudaMalloc(&rD, A1) );
//   TRY( cudaMalloc(&aD, A1) );
//   TRY( cudaMalloc(&eD, E1) );
//   TRY( cudaMemcpy(wD, w, W1, cudaMemcpyHostToDevice) );

//   fillKernel<<<blocks, threads>>>(rD, N, T(1.0/N));
//   while (1) {
//     fillKernel<<<1, 1>>>(eD, 1, 0);
//     pageRankKernel<<<blocks, threads>>>(eD, aD, rD, wD, N, p, E);
//     TRY( cudaMemcpy(&e, eD, E1, cudaMemcpyDeviceToHost) );
//     swap(aD, rD);
//     if (!e) break;
//   }
//   TRY( cudaMemcpy(a, rD, A1, cudaMemcpyDeviceToHost) );

//   TRY( cudaFree(wD) );
//   TRY( cudaFree(rD) );
//   TRY( cudaFree(aD) );
//   TRY( cudaFree(eD) );
// }


// template <class T>
// void pageRankCuda(T *a, T *w, int N, pageRankOptions<T> o=pageRankOptions<T>()) {
//   pageRankCuda(a, w, N, o.damping, o.convergence);
// }


// template <class T>
// void pageRankCuda(T *a, DenseDiGraph<T>& x, pageRankOptions<T> o=pageRankOptions<T>()) {
//   pageRankCuda(a, x.weights, x.order, o);
// }
