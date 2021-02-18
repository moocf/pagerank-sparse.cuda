#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <memory>
#include <omp.h>
#include "_cuda.h"
#include "add.h"
#include "fill.h"
#include "DiGraph.h"
#include "errorAbs.h"

using std::vector;
using std::unordered_map;
using std::unique_ptr;
using std::swap;
using std::max;




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
template <class K, class L, class M, class T>
void pageRankStep(DiGraph<K, L, M>& x, T p, unordered_map<K, T>& r, unordered_map<K, T>& a) {
  int N = x.order();
  fill(a, (1-p)/N);
  for (auto&& u : x.vertices()) {
    int d = x.degree(u);
    if (d > 0) addAt(a, x.edges(u), p*r[u]/d);
    else addAt(a, x.vertices(), p*r[u]/N);
  }
}


template <class K, class L, class M, class T>
auto& pageRank(DiGraph<K, L, M>& x, T p, T E) {
  int N = x.order();
  auto& r = *new unordered_map<K, T>(N);
  auto& a = *new unordered_map<K, T>(N);
  fillAt(r, x.vertices(), T(1)/N);
  while (1) {
    pageRankStep(x, p, r, a);
    T e = errorAbs(a, r);
    if (e < E) break;
    swap(a, r);
  }
  return a;
}


template <class K, class L, class M, class T>
void pageRankStep(IndexedDiGraph<K, L, M>& x, T p, vector<T>& r, vector<T>& a) {
  int N = x.order(), S = x.span();
  fill(a, (1-p)/N);
  for (int u=0; u<S; u++) {
    if (!x.hasVertex(u)) continue;
    int d = x.degree(u);
    if (d > 0) addAt(a, x.edges(u), p*r[u]/d);
    else add(a, p*r[u]/N);
  }
}


template <class K, class L, class M, class T>
auto& pageRank(IndexedDiGraph<K, L, M>& x, T p, T E) {
  int N = x.order(), S = x.span();
  auto& r = *new vector<T>(S);
  auto& a = *new vector<T>(S);
  fill(r, T(1)/N);
  while (1) {
    pageRankStep(x, p, r, a);
    T e = errorAbs(a, r);
    if (e < E) break;
    swap(a, r);
  }
  return a;
}


template <class K, class L, class M, class T>
void pageRankStep(CompactDiGraph<K, L, M>& x, T p, vector<T>& r, vector<T>& a) {
  int N = x.order();
  fill(a, (1-p)/N);
  for (int i=0; i<N; i++) {
    int d = x.degreeI(i);
    if (d > 0) addAt(a, x.edgesI(i), p*r[i]/d);
    else add(a, p*r[i]/N);
  }
}


template <class K, class L, class M, class T>
auto& pageRank(CompactDiGraph<K, L, M>& x, T p, T E) {
  int N = x.order();
  auto& r = *new vector<T>(N);
  auto& a = *new vector<T>(N);
  fill(r, T(1)/N);
  while (1) {
    pageRankStep(x, p, r, a);
    T e = errorAbs(a, r);
    if (e < E) break;
    swap(a, r);
  }
  return a;
}


template <class G, class T=float>
auto& pageRank(G& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRank(x, o.damping, o.convergence);
}




// template <class T>
// void pageRankPullStep(DiGraph& x, vector<T>& c, vector<T>& a) {
//   int S = x.span();
//   for (int i=0; i<S; i++) {
//     if (!x.hasVertex(i)) continue;
//     a[i] = sumAt(c, x.edges(i));
//   }
// }


// template <class T>
// void pageRankPullPostprocess(vector<T>& a, vector<T>& d, T p, T q) {
// }


// template <class T>
// vector<T>& pageRankPull(DiGraph& x, T p,T E) {
//   int S = x.span(), N = x.order();
//   vector<T>& r = *new vector<T>(S);
//   vector<T>& a = *new vector<T>(S);
//   fill(r, T(1)/N);
//   while (1) {
//   }
// }


template <class T>
__global__ void pageRankKernel(bool *has, int *deg, int *vout, int *eout, int S, int N, T p, T E) {
  DEFINE(t, b, B, G);
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
