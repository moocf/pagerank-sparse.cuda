#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <memory>
#include <omp.h>
#include "_cuda.h"
#include "DiGraph.h"
#include "measureDuration.h"
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
template <class G, class V, class T>
void pageRankPushStep(V& a, V& r, G& x, T p) {
  int N = x.order();
  fill(a, (1-p)/N);
  for (int u : x.vertices()) {
    int d = x.degree(u);
    if (d > 0) addAt(a, x.edges(u), p*r[u]/d);
    else add(a, p*r[u]/N);
  }
}


template <class G, class V, class T>
auto& pageRankPushCore(V& a, V& r, G& x, T p, T E) {
  T e0 = T();
  int N = x.order();
  fill(r, T(1)/N);
  while (1) {
    pageRankPushStep(a, r, x, p);
    T e = errorAbs(a, r);
    if (e < E || e == e0) break;
    swap(a, r);
    e0 = e;
  }
  fillAt(a, x.nonVertices(), T());
  return a;
}


template <class G, class T>
auto pageRankPush(float& t, G& x, T p, T E) {
  auto a = x.createVertexData(T());
  auto r = x.createVertexData(T());
  t = measureDuration([&]() { pageRankPushCore(a, r, x, p, E); });
  return a;
}

template <class G, class T=float>
auto pageRankPush(float& t, G& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankPush(t, x, o.damping, o.convergence);
}




template <class G, class V, class T>
T pageRankTeleport(G& x, V& r, T p, int N) {
  T a = (1-p)/N;
  for (auto u : x.vertices())
    if (x.vertexData(u) == 0) a += p*r[u]/N;
  return a;
}

template <class G, class V, class T>
void pageRankFactor(V& a, G& x, T p) {
  int N = x.order();
  for (auto u : x.vertices()) {
    int d = x.vertexData(u);
    a[u] = d>0? p/d : 0;
  }
}


template <class G, class V, class T>
void pageRankStep(V& a, V& c, G& x, T c0) {
  for (auto v : x.vertices())
    a[v] = c0 + sumAt(c, x.edges(v));
}


template <class G, class V, class T>
auto& pageRankCore(V& a, V& r, V& f, V& c, G& x, T p, T E) {
  T e0 = T();
  int N = x.order();
  fillAt(r, x.vertices(), T(1)/N);
  pageRankFactor(f, x, p);
  while (1) {
    T c0 = pageRankTeleport(x, r, p, N);
    multiply(c, r, f);
    pageRankStep(a, c, x, c0);
    T e = errorAbs(a, r);
    if (e < E || e == e0) break;
    swap(a, r);
    e0 = e;
  }
  fillAt(a, x.nonVertices(), T());
  return a;
}

template <class G, class T>
auto pageRank(float& t, G& x, T p, T E) {
  auto a = x.createVertexData(T());
  auto r = x.createVertexData(T());
  auto f = x.createVertexData(T());
  auto c = x.createVertexData(T());
  t = measureDuration([&]() { pageRankCore(a, r, f, c, x, p, E); });
  return a;
}

template <class G, class T=float>
auto pageRank(float& t, G& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRank(t, x, o.damping, o.convergence);
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
__global__ void pageRankKernelStep(T *a, T *c, int *vfrom, int *efrom, T c0, int N) {
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


template <class T>
T* pageRankCudaCore(T *e, T *r0, T *a, T *f, T *r, T *c, int *vfrom, int *efrom, int *vdata, int N, T p, T E) {
  int threads = _THREADS;
  int blocks = min(ceilDiv(N, threads), _BLOCKS);
  int B1 = blocks * sizeof(T);
  T eH[_BLOCKS], r0H[_BLOCKS], e0 = T();
  fillKernel<<<blocks, threads>>>(r, N, T(1)/N);
  pageRankFactorKernel<<<blocks, threads>>>(f, vdata, p, N);
  while (1) {
    sumIfNotKernel<<<blocks, threads>>>(r0, r, vdata, N);
    TRY( cudaMemcpy(r0H, r0, B1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0H, blocks)/N;
    multiplyKernel<<<blocks, threads>>>(c, r, f, N);
    pageRankKernelStep<<<blocks, threads>>>(a, c, vfrom, efrom, c0, N);
    errorAbsKernel<<<blocks, threads>>>(e, r, a, N);
    TRY( cudaMemcpy(eH, e, B1, cudaMemcpyDeviceToHost) );
    T f = sum(eH, blocks);
    if (f < E || f == e0) break;
    swap(a, r);
    e0 = f;
  }
  return a;
}


template <class G, class T>
auto pageRankCuda(float& t, G& x, T p, T E) {
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
  vector<T> a(N);

  T *eD, *r0D, *fD, *rD, *cD, *aD, *bD;
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

  t = measureDuration([&]() { bD = pageRankCudaCore(eD, r0D, aD, fD, rD, cD, vfromD, efromD, vdataD, N, p, E); });
  TRY( cudaMemcpy(a.data(), bD, N1, cudaMemcpyDeviceToHost) );

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
auto pageRankCuda(float& t, G& x, pageRankOptions<T> o=pageRankOptions<T>()) {
  return pageRankCuda(t, x, o.damping, o.convergence);
}
