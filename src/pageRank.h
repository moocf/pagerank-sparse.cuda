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
#include <stdio.h>

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
  auto vdata = x.vertexData();
  transform(vdata.begin(), vdata.end(), a.begin(), [=](int d) { return d>0? p/d : 0; });
}


template <class G, class T>
void pageRankPullStep(vector<T>& a, vector<T>& c, G& x, T c0) {
  for (int v : x.vertices())
    a[v] = c0 + sumAt(c, x.edges(v));
}


template <class G, class T>
vector<T>& pageRankPull(G& x, T p, T E) {
  int N = x.order(), S = x.span();
  auto& r = *new vector<T>(S);
  auto& f = *new vector<T>(S);
  auto& c = *new vector<T>(S);
  auto& a = *new vector<T>(S);
  fillAt(r, x.vertices(), T(1)/N);
  pageRankFactor(f, x, p);
  while (1) {
    T c0 = pageRankTeleport(x, r, p, N);
    multiply(c, r, f);
    pageRankPullStep(a, c, x, c0);
    T e = errorAbs(a, r);
    if (e < E) break;
    swap(a, r);
  }
  return a;
}

template <class G, class T=float>
auto& pageRankPull(G& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankPull(x, o.damping, o.convergence);
}




template <class T, class V>
__device__ void pageRankFactorKernelLoop(T *a, V *vdata, T p, int N, int i, int DI) {
  for (; i<N; i+=DI) {
    V d = vdata[i];
    a[i] = d>0? p/d : 0;
  }
}

template <class T, class V>
__global__ void pageRankFactorKernel(T *a, V *vdata, T p, int N) {
  DEFINE(t, b, B, G);
  pageRankFactorKernelLoop(a, vdata, p, N, B*b+t, G*B);
}


template <class T>
__global__ void pageRankPullKernelStep(T *a, T *c, int *vfrom, int *efrom, T c0, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[_THREADS];

  for (int v=b; v<N; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t == 0) a[v] = c0 + cache[0];
  }
}


template <class G, class T>
vector<T> pageRankPullCuda(G& x, T p, T E) {
  int N = x.order();
  auto vfrom = x.sourceOffsets();
  auto efrom = x.destinationIndices();
  auto vdata = x.vertexData();
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), _BLOCKS);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = N  * sizeof(int);
  int B1 = blocks * sizeof(T);
  int N1 = N      * sizeof(T);

  vector<T> r0(blocks);
  vector<T> e(blocks);
  vector<T> a(N);

  T *eD, *r0D, *fD, *rD, *cD, *aD;
  int *vfromD, *efromD, *vdataD;
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMalloc(&r0D, B1) );
  TRY( cudaMalloc(&eD,  B1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );
  printf("VFROM1: %d\n", VFROM1);
  printf("EFROM1: %d\n", EFROM1);
  printf("VDATA1: %d\n", VDATA1);
  printf("B1: %d\n", B1);
  printf("N1: %d\n", N1);

  fillKernel<<<blocks, threads>>>(rD, N, T(1)/N);
  TRY( cudaMemcpy(r0.data(), r0D, B1, cudaMemcpyDeviceToHost) );
  pageRankFactorKernel<<<blocks, threads>>>(fD, vdataD, p, N);
  TRY( cudaMemcpy(r0.data(), r0D, B1, cudaMemcpyDeviceToHost) );
  while (1) {
    printf("pageRankPullCuda\n");
    sumIfNotKernel<<<blocks, threads>>>(r0D, rD, vdataD, N);
    TRY( cudaMemcpy(r0.data(), r0D, B1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0)/N;
    multiplyKernel<<<blocks, threads>>>(cD, rD, fD, N);
    pageRankPullKernelStep<<<blocks, threads>>>(aD, cD, vfromD, efromD, c0, N);
    errorAbsKernel<<<blocks, threads>>>(eD, rD, aD, N);
    TRY( cudaMemcpy(e.data(), eD, B1, cudaMemcpyDeviceToHost) );
    if (sum(e) < E) break;
    swap(aD, rD);
  }
  TRY( cudaMemcpy(a.data(), aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(eD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(aD) );
  return a;
}

template <class G, class T=float>
auto pageRankPullCuda(G& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankPullCuda(x, o.damping, o.convergence);
}
