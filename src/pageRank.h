#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <memory>
#include <omp.h>
#include "_cuda.h"
#include "DiGraph.h"
#include "ceilDiv.h"
#include "measureDuration.h"
#include "count.h"
#include "add.h"
#include "sum.h"
#include "dotProduct.h"
#include "fill.h"
#include "multiply.h"
#include "errorAbs.h"
#include "vertices.h"
#include "sourceOffsets.h"
#include "destinationIndices.h"
#include "vertexData.h"
#include "vertexContainer.h"

using std::vector;
using std::unordered_map;
using std::unique_ptr;
using std::lower_bound;
using std::swap;
using std::max;




enum struct PageRankMode {
  BLOCK,
  THREAD,
  DYNAMIC,
  SWITCHED
};


template <class T>
struct PageRankOptions {
  typedef PageRankMode Mode;
  Mode mode;
  T damping;
  T convergence;

  PageRankOptions(Mode _mode=Mode::BLOCK, T _damping=0.85, T _convergence=1e-6) {
    mode = _mode;
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
  auto a = x.vertexContainer(T());
  auto r = x.vertexContainer(T());
  t = measureDuration([&]() { pageRankPushCore(a, r, x, p, E); });
  return a;
}

template <class G, class T=float>
auto pageRankPush(float& t, G& x, PageRankOptions<T> o=PageRankOptions<T>()) {
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
  auto a = x.vertexContainer(T());
  auto r = x.vertexContainer(T());
  auto f = x.vertexContainer(T());
  auto c = x.vertexContainer(T());
  t = measureDuration([&]() { pageRankCore(a, r, f, c, x, p, E); });
  return a;
}

template <class G, class T=float>
auto pageRank(float& t, G& x, PageRankOptions<T> o=PageRankOptions<T>()) {
  return pageRank(t, x, o.damping, o.convergence);
}




template <class T, class V>
__global__ void pageRankFactorKernel(T *a, V *vdata, T p, int N) {
  DEFINE(t, b, B, G);
  for (int i=B*b+t, DI=G*B; i<N; i+=DI) {
    V d = vdata[i];
    a[i] = d>0? p/d : 0;
  }
}


template <class T>
__global__ void pageRankBlockKernel(T *a, T *c, int *vfrom, int *efrom, T c0, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  for (int v=b; v<N; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t == 0) a[v] = c0 + cache[0];
  }
}


template <class T>
__global__ void pageRankThreadKernel(T *a, T *c, int *vfrom, int *efrom, T c0, int N) {
  DEFINE(t, b, B, G);

  for (int v=B*b+t; v<N; v+=G*B) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}


template <class T>
__global__ void pageRankDynamicKernel(T *a, T *c, int *vfrom, int *efrom, T c0, int N) {
  DEFINE(t, b, B, G);

  for (int v=B*b+t; v<N; v+=G*B) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    if (ideg < B/2) a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
    else pageRankBlockKernel<<<1, B>>>(&a[v], c, &vfrom[v], efrom, c0, 1);
  }
}


template <class T>
void pageRankKernelCall(int G, int B, T *a, T *c, int *vfrom, int *efrom, T c0, int N, PageRankMode M, int S) {
  typedef PageRankMode Mode;
  switch (M) {
    default:
    case Mode::BLOCK:   pageRankBlockKernel<<<G, B>>>(a, c, vfrom, efrom, c0, N); break;
    case Mode::THREAD:  pageRankThreadKernel<<<G, B>>>(a, c, vfrom, efrom, c0, N); break;
    // case Mode::DYNAMIC: pageRankDynamicKernel<<<G, B>>>(a, c, vfrom, efrom, c0, N); break;
    case Mode::SWITCHED:
      pageRankThreadKernel<<<G, B>>>(a, c, vfrom, efrom, c0, S);
      pageRankBlockKernel<<<G, B>>>(a+S, c, vfrom+S, efrom, c0, N-S);
      break;
  }
}




template <class T>
T* pageRankCudaCore(T *e, T *r0, T *a, T *f, T *r, T *c, int *vfrom, int *efrom, int *vdata, int N, PageRankMode M, T p, T E, int S) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  int G1 = G * sizeof(T);
  T eH[GRID_DIM], r0H[GRID_DIM], e0 = T();
  fillKernel<<<G, B>>>(r, N, T(1)/N);
  pageRankFactorKernel<<<G, B>>>(f, vdata, p, N);
  while (1) {
    sumIfNotKernel<<<G, B>>>(r0, r, vdata, N);
    TRY( cudaMemcpy(r0H, r0, G1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0H, G)/N;
    multiplyKernel<<<G, B>>>(c, r, f, N);
    pageRankKernelCall(G, B, a, c, vfrom, efrom, c0, N, M, S);
    errorAbsKernel<<<G, B>>>(e, r, a, N);
    TRY( cudaMemcpy(eH, e, G1, cudaMemcpyDeviceToHost) );
    T f = sum(eH, G);
    if (f < E || f == e0) break;
    swap(a, r);
    e0 = f;
  }
  return a;
}

// template <class T>
// T* pageRankCudaCore(T *e, T *r0, T *a, T *f, T *r, T *c, int *vfrom, int *efrom, int *vdata, int N, PageRankMode M, T p, T E, int S) {
//   int B = BLOCK_DIM;
//   int G = min(ceilDiv(N, B), GRID_DIM);
//   int B1 = G * sizeof(T);
//   T eH[GRID_DIM], r0H[GRID_DIM], e0 = T();
//   fillKernel<<<G, B, 0, s1>>>(r, N, T(1)/N);
//   pageRankFactorKernel<<<G, B, 0, s2>>>(f, vdata, p, N);
//   while (1) {
//     sumIfNotKernel<<<G, B, 0, s1>>>(r0, r, vdata, N);
//     multiplyKernel<<<G, B, 0, s2>>>(c, r, f, N);
//     TRY( cudaMemcpy(r0H, r0, B1, cudaMemcpyDeviceToHost) );
//     T c0 = (1-p)/N + p*sum(r0H, G)/N;
//     pageRankKernelCall(G, B, a, c, vfrom, efrom, c0, N, M, S);
//     errorAbsKernel<<<G, B, 0, s3>>>(e, r, a, N);
//     TRY( cudaMemcpy(eH, e, B1, cudaMemcpyDeviceToHost) );
//     T f = sum(eH, G);
//     if (f < E || f == e0) break;
//     swap(a, r);
//     e0 = f;
//   }
//   return a;
// }


template <class G>
auto pageRankVertices(G& x, PageRankMode M) {
  using K = typename G::TKey;
  typedef PageRankMode Mode;
  if (M != Mode::SWITCHED) return vertices(x);
  return verticesBy(x, [&](K u) { return x.degree(u); });
}

template <class G, class K>
int pageRankSwitchPoint(G& x, vector<K>& ks, PageRankMode M) {
  typedef PageRankMode Mode;
  if (M != Mode::SWITCHED) return 0;
  int deg = int(0.5 * BLOCK_DIM);
  auto it = lower_bound(ks.begin(), ks.end(), deg, [&](K u, int d) {
    return x.degree(u) < d;
  });
  return it - ks.begin();
}


template <class G, class T>
auto pageRankCuda(float& t, G& x, PageRankMode M, T p, T E) {
  using K = typename G::TKey;
  auto ks = pageRankVertices(x, M);
  auto vfrom = sourceOffsets(x, ks);
  auto efrom = destinationIndices(x, ks);
  auto vdata = vertexData(x, ks);  // outDegree
  int S = pageRankSwitchPoint(x, ks, M);
  int N = x.order();
  int B = BLOCK_DIM;
  int g = min(ceilDiv(N, B), GRID_DIM);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int G1 = g * sizeof(T);
  int N1 = N * sizeof(T);
  vector<T> a(N);

  T *eD, *r0D, *fD, *rD, *cD, *aD, *bD;
  int *vfromD, *efromD, *vdataD;
  // TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMalloc(&r0D, G1) );
  TRY( cudaMalloc(&eD,  G1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );

  t = measureDuration([&]() { bD = pageRankCudaCore(eD, r0D, aD, fD, rD, cD, vfromD, efromD, vdataD, N, M, p, E, S); });
  TRY( cudaMemcpy(a.data(), bD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(eD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(aD) );
  return vertexContainer(x, a, ks);
}

template <class G, class T=float>
auto pageRankCuda(float& t, G& x, PageRankOptions<T> o=PageRankOptions<T>()) {
  return pageRankCuda(t, x, o.mode, o.damping, o.convergence);
}
