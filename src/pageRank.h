#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <utility>
#include <memory>
#include <omp.h>
#include "_cuda.h"
#include "ceilDiv.h"
#include "measure.h"
#include "sgn.h"
#include "abs.h"
#include "sum.h"
#include "from.h"
#include "slice.h"
#include "fill.h"
#include "multiply.h"
#include "error.h"
#include "reorder.h"
#include "join.h"
#include "unique.h"
#include "vertices.h"
#include "edges.h"
#include "components.h"
#include "blockgraph.h"
#include "sort.h"
#include "chains.h"
#include "identicals.h"

using std::vector;
using std::unordered_map;
using std::lower_bound;
using std::swap;
using std::pow;
using std::min;
using std::abs;



// PAGE-RANK OPTIONS
// -----------------

enum struct PageRankMode {
  BLOCK,
  THREAD,
  SWITCHED
};

struct PageRankFlags {
  bool splitComponents  = false;
  bool largeComponents  = false;
  bool orderComponents  = false;
  bool orderVertices    = false;
  bool crossPropagate   = false;
  bool removeIdenticals = false;
  bool removeChains     = false;
  bool skipConverged    = false;

  PageRankFlags() = default;
  PageRankFlags(int n) {
    splitComponents  = n & 128;
    largeComponents  = n & 64;
    orderComponents  = n & 32;
    orderVertices    = n & 16;
    crossPropagate   = n & 8;
    removeIdenticals = n & 4;
    removeChains     = n & 2;
    skipConverged    = n & 1;
  }
};

template <class T>
struct PageRankOptions {
  typedef PageRankMode  Mode;
  typedef PageRankFlags Flags;
  Mode  mode;
  Flags flags;
  T damping;
  T convergence;
  int maxIterations;

  PageRankOptions(Mode _mode=Mode::BLOCK, Flags _flags={}, T _damping=0.85, T _convergence=1e-6, int _maxIterations=10000) {
    mode = _mode;
    flags = _flags;
    damping = _damping;
    convergence = _convergence;
    maxIterations = _maxIterations;
  }
};




// PAGE-RANK UPDATE (DYNAMIC)
// --------------------------

enum struct PageRankUpdateMode {
  RANDOM,
  DEGREE,
  RANK
};

struct PageRankUpdateFlags {
  bool addVertices    = false;
  bool removeVertices = false;
  bool addEdges       = false;
  bool removeEdges    = false;

  PageRankUpdateFlags() = default;
  PageRankUpdateFlags(int n) {
    addVertices    = n & 8;
    removeVertices = n & 4;
    addEdges       = n & 2;
    removeEdges    = n & 1;
  }
};

struct PageRankUpdateOptions {
  typedef PageRankUpdateMode  Mode;
  typedef PageRankUpdateFlags Flags;
  Mode mode;
  Flags flags;

  PageRankUpdateOptions(Mode _mode=Mode::RANDOM, Flags _flags={}) {
    mode = _mode;
    flags = _flags;
  }
};




// PAGE-RANK
// ---------

template <class G, class C, class T>
T pageRankTeleport(C& r, G& x, T p, int N) {
  T a = (1-p)/N;
  for (auto u : x.vertices())
    if (x.vertexData(u) == 0) a += p*r[u]/N;
  return a;
}

template <class G, class C, class T>
void pageRankFactor(C& a, G& x, T p) {
  int N = x.order();
  for (auto u : x.vertices()) {
    int d = x.vertexData(u);
    a[u] = d>0? p/d : 0;
  }
}


template <class G, class C, class T>
void pageRankOnce(C& a, C& c, G& x, T c0) {
  for (auto v : x.vertices())
    a[v] = c0 + sumAt(c, x.edges(v));
}


template <class G, class C, class T>
auto& pageRankLoop(float& m, C& a, C& r, C& f, C& c, G& x, T p, T E, int L) {
  T e0 = T();
  int N = x.order();
  fillAt(r, T(1)/N, x.vertices());
  pageRankFactor(f, x, p);
  int l = 0;
  for (; l<L; l++) {
    T c0 = pageRankTeleport(r, x, p, N);
    multiply(c, r, f);
    pageRankOnce(a, c, x, c0);
    T e1 = absError(a, r);
    if (e1 < E || e1 == e0) break;
    swap(a, r);
    e0 = e1;
  }
  m += l;
  return a;
}

template <class G, class C, class T>
auto& pageRankCore(float& m, C& a, C& r, C& f, C& c, G& x, T p, T E, int L) {
  int N = x.order(); m = 0;
  fillAt(r, T(1)/N, x.vertices());
  pageRankFactor(f, x, p);
  return pageRankLoop(m, a, r, f, c, x, p, E, L);
  // fillAt(b, x.nonVertices(), T());
}


template <class G, class H, class T=float>
auto pageRank(float& t, float& m, G& x, H& xt, PageRankOptions<T> o=PageRankOptions<T>()) {
  auto p = o.damping;
  auto E = o.convergence;
  auto L = o.maxIterations;
  auto a = xt.vertexContainer(T());
  auto r = xt.vertexContainer(T());
  auto f = xt.vertexContainer(T());
  auto c = xt.vertexContainer(T());
  t = measureDuration([&]() { pageRankCore(m, a, r, f, c, xt, p, E, L); });
  return a;
}




// PAGE-RANK HELPERS
// -----------------

template <class G, class K>
int pageRankSwitchPoint(G& xt, vector<K>& ks) {
  int deg = int(0.5 * BLOCK_DIM);
  auto it = lower_bound(ks.begin(), ks.end(), deg, [&](K u, int d) {
    return xt.degree(u) < d;
  });
  return it - ks.begin();
}




// PAGE-RANK KERNELS (CUDA)
// ------------------------

template <class T>
__global__ void pageRankFactorKernel(T *a, int *vdata, T p, int N) {
  DEFINE(t, b, B, G);
  for (int v=B*b+t, DV=G*B; v<N; v+=DV) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}


template <class T>
__global__ void pageRankBlockKernel(T *a, T *r, T *c, int *vfrom, int *efrom, T c0, int i, int n) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  for (int v=i+b, V=i+n; v<V; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t == 0) a[v] = c0 + cache[0];
  }
}


template <class T>
__global__ void pageRankThreadKernel(T *a, T *r, T *c, int *vfrom, int *efrom, T c0, int i, int n) {
  DEFINE(t, b, B, G);

  for (int v=i+B*b+t, V=i+n; v<V; v+=G*B) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}




// PAGE-RANK KERNEL CALLERS (CUDA)
// -------------------------------

template <class T>
void pageRankBlockKernelCall(T *a, T *r, T *c, int *vfrom, int *efrom, T c0, int i, int n) {
  int B = BLOCK_DIM;
  int G = min(n, GRID_DIM);
  pageRankBlockKernel<<<G, B>>>(a, r, c, vfrom, efrom, c0, i, n);
}


template <class T>
void pageRankThreadKernelCall(T *a, T *r, T *c, int *vfrom, int *efrom, T c0, int i, int n) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(n, B), GRID_DIM);
  pageRankThreadKernel<<<G, B>>>(a, r, c, vfrom, efrom, c0, i, n);
}


template <class T, class I>
void pageRankKernelWave(T *a, T *r, T *c, int *vfrom, int *efrom, T c0, int i, I&& ns) {
  for (int n : ns) {
    if (n > 0) pageRankBlockKernelCall (a, r, c, vfrom, efrom, c0, i, n);
    else       pageRankThreadKernelCall(a, r, c, vfrom, efrom, c0, i, -n);
    i += abs(n);
  }
}




// PAGE-RANK (CUDA)

template <class T, class I>
T* pageRankCudaLoop(float& m, T *e, T *r0, T *eD, T *r0D, T *aD, T *cD, T *rD, T *fD, int *vfromD, int *efromD, int *vdataD, int i, I&& ns, int N, T p, T E, int L) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  int G1 = G * sizeof(T), l = 1;
  for (; l<L; l++) {
    sumIfNotKernel<<<G, B>>>(r0D, rD, vdataD, N);
    multiplyKernel<<<G, B>>>(cD,  rD, fD,     N);
    TRY( cudaMemcpy(r0, r0D, G1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0, G)/N;
    pageRankKernelWave(aD, rD, cD, vfromD, efromD, c0, i, ns);
    absErrorKernel<<<G, B>>>(eD, rD, aD, N);
    TRY( cudaMemcpy(e, eD, G1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, G);
    if (e1 < E) break;
    swap(aD, rD);
  }
  m += l;
  return aD;
}


template <class T, class I>
T* pageRankCudaCore(float& m, T *e, T *r0, T *eD, T *r0D, T *aD, T *cD, T *rD, T *fD, int *vfromD, int *efromD, int *vdataD, I&& ns, int N, T p, T E, int L) {
  int B = BLOCK_DIM; m = 0;
  int G = min(ceilDiv(N, B), GRID_DIM);
  fillKernel<<<G, B>>>(rD, N, T(1)/N);
  pageRankFactorKernel<<<G, B>>>(fD, vdataD, p, N);
  return pageRankCudaLoop(m, e, r0, eD, r0D, aD, cD, rD, fD, vfromD, efromD, vdataD, 0, ns, N, p, E, L);
}


template <class G, class H, class T=float>
auto pageRankCuda(float& t, float& m, G& x, H& xt, PageRankOptions<T> o=PageRankOptions<T>()) {
  using K = typename G::TKey;
  auto p = o.damping;
  auto E = o.convergence;
  auto L = o.maxIterations;
  int  N = xt.order();
  auto ks    = verticesByDegree(xt);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  int  S     = pageRankSwitchPoint(xt, ks);
  vector<int> ns {-S, N-S};
  int g = GRID_DIM;
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int G1 = g * sizeof(T);
  int N1 = N * sizeof(T);
  vector<T> a(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD, *bD;
  int *vfromD, *efromD, *vdataD;
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&e,  G1, cudaHostAllocDefault) );
  TRY( cudaHostAlloc(&r0, G1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&eD,  G1) );
  TRY( cudaMalloc(&r0D, G1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );

  t = measureDuration([&]() { bD = pageRankCudaCore(m, e, r0, eD, r0D, aD, cD, rD, fD, vfromD, efromD, vdataD, ns, N, p, E, L); });
  TRY( cudaMemcpy(a.data(), bD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(eD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(vdataD) );
  TRY( cudaFreeHost(e) );
  TRY( cudaFreeHost(r0) );
  return vertexContainer(xt, a, ks);
}
