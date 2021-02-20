#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <memory>
#include <omp.h>
#include "_cuda.h"
#include "DiGraph.h"
#include "count.h"
#include "add.h"
#include "sum.h"
#include "fill.h"
#include "multiply.h"
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
template <class G, class T>
void pageRankStep(vector<T>& a, vector<T>& r, G& x, T p) {
  int N = x.order();
  fill(a, (1-p)/N);
  for (int u : x.vertices()) {
    int d = x.degree(u);
    if (d > 0) addAt(a, x.edges(u), p*r[u]/d);
    else add(a, p*r[u]/N);
  }
}


template <class G, class T>
vector<T>& pageRank(G& x, T p, T E) {
  int N = x.order(), S = x.span();
  auto& a = *new vector<T>(S);
  auto& r = *new vector<T>(S);
  fill(r, T(1)/N);
  while (1) {
    pageRankStep(a, r, x, p);
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




template <class G, class T>
T pageRankTeleport(G& x, vector<T>& r, T p, int N) {
  T a = (1-p)/N;
  for (int u : x.vertices())
    if (x.vertexData(u) == 0) a += p*r[u]/N;
  return a;
}

template <class G, class T>
void pageRankFactor(vector<T>& a, G& x, T p) {
  int N = x.order();
  auto& vdata = x.vertexData();
  transform(vdata.begin(), vdata.end(), a.begin(), [=](int d) { return d>0? p/d : 0; });
}


template <class G, class T>
void pageRankPullStep(vector<T>& a, vector<T>& c, G& x, T p, T c0) {
  for (int v : x.vertices())
    a[v] = c0 + sumAt(c, x.edges(v));
}


template <class G, class T>
vector<T>& pageRankPull(G& x, T p, T E) {
  int N = x.order(), S = x.span();
  // int Z = count(x.vertexData(), 0)-(S-N);
  // printf("Z: %d\n", Z);
  T r0 = T(1)/N;
  auto& r = *new vector<T>(S);
  auto& f = *new vector<T>(S);
  auto& c = *new vector<T>(S);
  auto& a = *new vector<T>(S);
  fillAt(r, x.vertices(), r0);
  pageRankFactor(f, x, p);
  while (1) {
    T c0 = pageRankTeleport(x, r, p, N);
    multiply(c, r, f);
    pageRankPullStep(a, c, x, p, c0);
    T e = errorAbs(a, r);
    if (e < E) break;
    swap(a, r);
    r0 = c0;
  }
  return a;
}

template <class G, class T=float>
auto& pageRankPull(G& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankPull(x, o.damping, o.convergence);
}




template <class T>
__global__ void pageRankKernel(T *a, T *c, int *vfrom, int *efrom, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  for (int v=b; v<N; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    a[v] = cache[0];
  }
}


template <class G, class T>
vector<T>& pageRankCuda(G& x, vector<int>& odeg, T p, T E) {
  int N = x.order(), M = x.size(), S = x.span();
  int threads = _THREADS;
  int blocks = max(ceilDiv(N, threads), 1024);
  int E1 = blocks * sizeof(T);
  int A1 = S * sizeof(T);
  int VFROM1 = x.sourceOffsets().size() * sizeof(int);
  int EFROM1 = x.destinationIndices().size() * sizeof(int);

  int Z = count(odeg, 0);
  vector<T> e(blocks);
  vector<T> f(S);
  transform(odeg, f, [=](int d) { return d>0? p/d : p/N; });

  T *eD, *fD, *rD, *cD, *aD;
  int *vfromD, *efromD;
  TRY( cudaMalloc(&eD, E1) );
  TRY( cudaMalloc(&fD, A1) );
  TRY( cudaMalloc(&rD, A1) );
  TRY( cudaMalloc(&cD, A1) );
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMemcpy(fD,      f.data(),               A1,     cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vfromD, &x.sourceOffsets(),      VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, &x.destinationIndices(), EFROM1, cudaMemcpyHostToDevice) );

  T r0 = T(1)/N;
  fillKernel<<<blocks, threads>>>(rD, N, r0);
  while (1) {
    T ct = pageRankTeleport(r0, p, N, Z);
    multiplyKernel<<<blocks, threads>>>(cD, rD, fD, S);
    pageRankKernel<<<blocks, threads>>>(aD, cD, vfromD, efromD, S);
    addKernel<<<blocks, threads>>>(aD, S, ct);
    errorAbsKernel<<<blocks, threads>>>(eD, rD, aD);
    TRY( cudaMemcpy(e.data(), eD, E1, cudaMemcpyDeviceToHost) );
    if (sum(e) < E) break;
    swap(aD, rD);
    r0 = ct;
  }
  auto& a = *new vector<T>(S);
  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(eD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  return a;
}

template <class G, class T=float>
auto pageRankCuda(G& x, vector<int>& odeg, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankCuda(x, odeg, o.damping, o.convergence);
}
