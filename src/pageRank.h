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
  typedef PageRankMode Mode;
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

template <class G, class C, class D>
void pageRankSortComponents(D& a, G& xt, C& ch, C& id) {
  auto chs = setFrom(ch, 1);
  auto ids = setFrom(id, 1);
  for (auto& c : a) {
    sort(c.begin(), c.end(), [&](auto u, auto v) {
      int du = ids.count(u) || chs.count(u)? -ids.count(u) : xt.degree(u);
      int dv = ids.count(v) || chs.count(v)? -ids.count(v) : xt.degree(v);
      return du < dv;
    });
  }
}


template <class G, class H, class C>
auto pageRankComponents(G& x, H& xt, C& cs, C& ch, C& id, PageRankMode M, PageRankFlags F) {
  using K = typename G::TKey;
  typedef PageRankMode Mode;
  vector<vector<K>> a;
  int n0 = F.largeComponents? GRID_DIM * BLOCK_DIM : 0;
  if (F.splitComponents) a = joinUntilSize(cs, n0);
  else a.push_back(vertices(x));
  if (F.orderVertices || M == Mode::SWITCHED) pageRankSortComponents(a, xt, ch, id);
  if (F.orderComponents) {
    auto b = blockgraph(x, a);
    auto bks = sort(b);
    reorder(a, bks);
  }
  return a;
}


template <class K>
auto pageRankVertexRoot(vector<K>& ks, vector<vector<K>>& ch, vector<vector<K>>& id) {
  vector<int> a(ks.size());
  fill(a, -1);
  auto is = indexMapFrom(ks);
  for (auto& vs : id) {
    auto u = vs[0];
    for (auto v : slice(vs, 1))
      a[is[v]] = is[u];
  }
  for (auto& vs : ch) {
    auto u = vs[0];
    for (auto v : slice(vs, 1))
      a[is[v]] = is[u];
  }
  return a;
}


template <class K>
auto pageRankVertexDistance(vector<K>& ks, vector<vector<K>>& ch) {
  vector<int> a(ks.size());
  auto is = indexMapFrom(ks);
  for (auto& vs : ch) {
    auto u = vs[0]; int i = 0;
    for (auto v : slice(vs, 1))
      a[is[v]] = ++i;
  }
  return a;
}


void pageRankMarkSpecial(vector<int>& a, vector<int>& vroot) {
  for (int i=0, I=vroot.size(); i<I; i++)
    if (vroot[i] >= 0) a[i] = -a[i];
}


template <class G, class K>
int pageRankSwitchPoint(G& xt, vector<K>& ks) {
  int deg = int(0.5 * BLOCK_DIM);
  auto it = lower_bound(ks.begin(), ks.end(), deg, [&](K u, int d) {
    return xt.degree(u) < d;
  });
  return it - ks.begin();
}


template <class C>
void pageRankAddStep(C& a, int n) {
  if (a.empty() || sgn(a.back()) != sgn(n)) a.push_back(n);
  else a.back() += n;
}

template <class G, class K, class C>
void pageRankStep(C& a, G& xt, vector<K>& ks, PageRankMode M) {
  typedef PageRankMode Mode;
  int n = ks.size();
  switch (M) {
    case Mode::BLOCK:  pageRankAddStep(a, n);  break;
    case Mode::THREAD: pageRankAddStep(a, -n); break;
    case Mode::SWITCHED:
      int s = pageRankSwitchPoint(xt, ks);
      if (s)   pageRankAddStep(a, -s);
      if (n-s) pageRankAddStep(a, n-s);
  }
}

template <class G, class K>
auto pageRankStep(G& xt, vector<K>& ks, PageRankMode M) {
  vector<int> a; pageRankStep(a, xt, ks, M);
  return a;
}


template <class G, class K>
auto pageRankWave(G& xt, vector<vector<K>>& cs, PageRankMode M) {
  vector<int> a;
  for (auto& c : cs)
    pageRankStep(a, xt, c, M);
  return a;
}




// PAGE-RANK KERNELS (CUDA)
// ------------------------

template <class T>
__device__ bool pageRankKernelIsConverged(T *a, T *r, bool fSC, int v) {
  return fSC && a[v] == r[v];
}

__device__ bool pageRankKernelIsVertexSpecial(int *vfrom, int v) {
  return vfrom[v] < 0;
}

template <class T>
__device__ T pageRankKernelCalculate(T r, int d, T c0, T p) {
  T pd = pow(p, d);
  return ((1-pd)/(1-p)) * c0 + pd * r;
}


template <class T>
__global__ void pageRankFactorKernel(T *a, int *vdata, T p, int N) {
  DEFINE(t, b, B, G);
  for (int v=B*b+t, DV=G*B; v<N; v+=DV) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}


template <class T>
__global__ void pageRankSpecialKernel(T *a, T *r, int *vroot, int *vdist, T c0, T p, int i, int n) {
  DEFINE(t, b, B, G);
  for (int v=i+B*b+t, V=i+n, DV=G*B; v<V; v+=DV) {
    if (vroot[v] < 0) continue;
    if (!vdist[v]) a[v] = a[vroot[v]];
    else a[v] = pageRankKernelCalculate(r[vroot[v]], vdist[v], c0, p);
  }
}


template <class T>
__global__ void pageRankBlockKernel(T *a, T *q, T *r, T *c, int *vfrom, int *efrom, T c0, bool fSC, int i, int n) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  for (int v=i+b, V=i+n; v<V; v+=G) {
    if (pageRankKernelIsConverged(a, r, fSC, v)) continue;
    if (pageRankKernelIsVertexSpecial(vfrom, v)) continue;
    int ebgn = abs(vfrom[v]);
    int ideg = abs(vfrom[v+1])-abs(vfrom[v]);
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t == 0) a[v] = c0 + (q? q[v]:0) + cache[0];
  }
}


template <class T>
__global__ void pageRankThreadKernel(T *a, T *q, T *r, T *c, int *vfrom, int *efrom, T c0, bool fSC, int i, int n) {
  DEFINE(t, b, B, G);

  for (int v=i+B*b+t, V=i+n; v<V; v+=G*B) {
    if (pageRankKernelIsConverged(a, r, fSC, v)) continue;
    if (pageRankKernelIsVertexSpecial(vfrom, v)) continue;
    int ebgn = abs(vfrom[v]);
    int ideg = abs(vfrom[v+1])-abs(vfrom[v]);
    a[v] = c0 + (q? q[v]:0) + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}




// PAGE-RANK KERNEL CALLERS (CUDA)
// -------------------------------

template <class T>
void pageRankBlockKernelCall(T *a, T *q, T *r, T *c, int *vfrom, int *efrom, T c0, bool fSC, int i, int n) {
  int B = BLOCK_DIM;
  int G = min(n, GRID_DIM);
  pageRankBlockKernel<<<G, B>>>(a, q, r, c, vfrom, efrom, c0, fSC, i, n);
}


template <class T>
void pageRankThreadKernelCall(T *a, T *q, T *r, T *c, int *vfrom, int *efrom, T c0, bool fSC, int i, int n) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(n, B), GRID_DIM);
  pageRankThreadKernel<<<G, B>>>(a, q, r, c, vfrom, efrom, c0, fSC, i, n);
}


template <class T, class I>
void pageRankKernelWave(T *a, T *q, T *r, T *c, int *vfrom, int *efrom, T c0, bool fSC, int i, I&& ns) {
  for (int n : ns) {
    if (n > 0) pageRankBlockKernelCall (a, q, r, c, vfrom, efrom, c0, fSC, i, n);
    else       pageRankThreadKernelCall(a, q, r, c, vfrom, efrom, c0, fSC, i, -n);
    i += abs(n);
  }
}




// PAGE-RANK (CUDA)

template <class T, class I>
T* pageRankCudaLoop(float& m, T *e, T *r0, T *eD, T *r0D, T *aD, T *cD, T *rD, T *fD, T *qD, int *vfromD, int *efromD, int *vdataD, int *vrootD, int *vdistD, int i, I&& ns, int n, int N, T p, T E, int L, bool fSC) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(n, B), GRID_DIM);
  int H = min(ceilDiv(N, B), GRID_DIM);
  int G1 = G * sizeof(T);
  int H1 = H * sizeof(T);
  int SKIP = 4;
  T e0 = T();
  int l = 0;
  for (; l<L; l++) {
    bool _fSC = fSC && (l%SKIP > 0);
    sumIfNotKernel<<<H, B>>>(r0D,  rD,   vdataD, N);
    multiplyKernel<<<G, B>>>(cD+i, rD+i, fD+i,   n);
    TRY( cudaMemcpy(r0, r0D, H1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0, G)/N;
    pageRankKernelWave(aD, qD, rD, cD, vfromD, efromD, c0, _fSC, i, ns);
    if (vrootD) pageRankSpecialKernel<<<G, B>>>(aD, rD, vrootD, vdistD, c0, p, i, n);
    absErrorKernel<<<G, B>>>(eD, rD, aD, N);
    TRY( cudaMemcpy(e, eD, G1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, G);
    if (e1 < E || e1 == e0) break;
    swap(aD, rD);
    e0 = e1;
  }
  m += (l*n)/N;
  return aD;
}


template <class T, class I>
T* pageRankCudaCore(float& m, T *e, T *r0, T *eD, T *r0D, T *aD, T *cD, T *rD, T *fD, int *vfromD, int *efromD, int *vdataD, int *vrootD, int *vdistD, I&& ns, int N, T p, T E, int L, bool fSC) {
  int B = BLOCK_DIM; m = 0;
  int G = min(ceilDiv(N, B), GRID_DIM);
  fillKernel<<<G, B>>>(rD, N, T(1)/N);
  pageRankFactorKernel<<<G, B>>>(fD, vdataD, p, N);
  return pageRankCudaLoop(m, e, r0, eD, r0D, aD, cD, rD, fD, (T*) nullptr, vfromD, efromD, vdataD, vrootD, vdistD, 0, ns, N, N, p, E, L, fSC);
}


template <class G, class H, class C, class T=float>
auto pageRankCuda(float& t, float& m, G& x, H& xt, C& xcs, C& xid, C& xch, PageRankOptions<T> o=PageRankOptions<T>()) {
  using K = typename G::TKey;
  vector<vector<K>> ks0;
  auto M = o.mode;
  auto F = o.flags;
  auto p = o.damping;
  auto E = o.convergence;
  auto L = o.maxIterations;
  bool fSC = F.skipConverged;
  bool fRI = F.removeIdenticals;
  bool fRC = F.removeChains;
  auto& id = fRI? xid : ks0;
  auto& ch = fRC? xch : ks0;
  auto cs = pageRankComponents(x, xt, xcs, ch, id, M, F);
  auto ns = pageRankWave(xt, cs, M);
  auto ks = join(cs);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);  // outDegree
  auto vroot = pageRankVertexRoot(ks, ch, id);
  auto vdist = pageRankVertexDistance(ks, ch);
  pageRankMarkSpecial(vfrom, vroot);
  // printf("isUnique(ch)?: %d\n", isUnique(ch));
  // printf("isUnique(id)?: %d\n", isUnique(id));
  int N = xt.order();
  int g = GRID_DIM;
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int VROOT1 = vroot.size() * sizeof(int);
  int VDIST1 = vdist.size() * sizeof(int);
  int G1 = g * sizeof(T);
  int N1 = N * sizeof(T);
  vector<T> a(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD, *bD;
  int *vfromD, *efromD, *vdataD;
  int *vrootD = NULL, *vdistD = NULL;
  cudaStream_t s1, s2, s3;
  // TRY( cudaProfilerStart() );
  TRY( cudaStreamCreate(&s1) );
  TRY( cudaStreamCreate(&s2) );
  TRY( cudaStreamCreate(&s3) );
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
  if (fRI || fRC) TRY( cudaMalloc(&vrootD, VROOT1) );
  if (fRI || fRC) TRY( cudaMalloc(&vdistD, VDIST1) );
  TRY( cudaMemcpyAsync(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaMemcpyAsync(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaMemcpyAsync(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice, s1) );
  if (fRI || fRC) TRY( cudaMemcpyAsync(vrootD, vroot.data(), VROOT1, cudaMemcpyHostToDevice, s1) );
  if (fRI || fRC) TRY( cudaMemcpyAsync(vdistD, vdist.data(), VDIST1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaStreamSynchronize(s1) );

  t = measureDuration([&]() { bD = pageRankCudaCore(m, e, r0, eD, r0D, aD, cD, rD, fD, vfromD, efromD, vdataD, vrootD, vdistD, ns, N, p, E, L, fSC); });
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
  if (fRI || fRC) TRY( cudaFree(vrootD) );
  if (fRI || fRC) TRY( cudaFree(vdistD) );
  TRY( cudaFreeHost(e) );
  TRY( cudaFreeHost(r0) );
  TRY( cudaStreamDestroy(s1) );
  TRY( cudaStreamDestroy(s2) );
  TRY( cudaStreamDestroy(s3) );
  // TRY( cudaProfilerStop() );
  return vertexContainer(xt, a, ks);
}

template <class G, class H, class T=float>
auto pageRankCuda(float& t, float& m, G& x, H& xt, PageRankOptions<T> o=PageRankOptions<T>()) {
  auto cs = components(x, xt);
  auto id = inIdenticals(x, xt);
  auto ch = chains(x, xt);
  return pageRankCuda(t, m, x, xt, cs, id, ch, o);
}




// UNUSED

/*
template <class T>
T* pageRankCudaLoopStreamed(int G, int B, cudaStream_t s1, cudaStream_t s2, cudaStream_t s3, T *e, T *r0, T *eD, T *r0D, T *aD, T *fD, T *rD, T *cD, int *vconvD, int *vfromD, int *efromD, int *vdataD, int N, int i, PageRankMode M, T p, T E, int S) {
  T e0 = T();
  int G1 = G * sizeof(T);
  sumIfNotKernel<<<G, B, 0, s1>>>(r0D,  rD+i, vdataD+i, N);
  multiplyKernel<<<G, B, 0, s2>>>(cD+i, rD+i, fD+i, N);
  while (1) {
    TRY( cudaStreamSynchronize(s1) );
    TRY( cudaStreamSynchronize(s2) );
    T c0 = (1-p)/N + p*sum(r0, G)/N;

    pageRankKernelCallStreamed(G, B, s1, s2, aD+i, cD, vconvD+i, vfromD+i, efromD, c0, N, M, S);
    TRY( cudaStreamSynchronize(s1) );
    TRY( cudaStreamSynchronize(s2) );
    swap(aD, rD);

    errorAbsKernel<<<G, B, 0, s3>>>(eD,   rD+i, aD+i, N);
    sumIfNotKernel<<<G, B, 0, s1>>>(r0D,  rD+i, vdataD+i, N);
    multiplyKernel<<<G, B, 0, s2>>>(cD+i, rD+i, fD+i, N);
    TRY( cudaMemcpyAsync(e,  eD,  G1, cudaMemcpyDeviceToHost, s3) );
    TRY( cudaMemcpyAsync(r0, r0D, G1, cudaMemcpyDeviceToHost, s1) );
    TRY( cudaStreamSynchronize(s3) );
    T e1 = sum(e, G);
    if (e1 < E || e1 == e0) break;
    e0 = e1;
  }
  return rD;
}

template <class T>
T* pageRankCudaCoreStreamed(cudaStream_t s1, cudaStream_t s2, cudaStream_t s3, T *e, T *r0, T *eD, T *r0D, T *aD, T *fD, T *rD, T *cD, int *vfromD, int *efromD, int *vdataD, int N, PageRankMode M, T p, T E, int S) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  pageRankFactorKernel<<<G, B, 0, s1>>>(fD, vdataD, p, N);
  fillKernel<<<G, B, 0, s2>>>(rD, N, T(1)/N);
  TRY( cudaStreamSynchronize(s1) );
  TRY( cudaStreamSynchronize(s2) );
  return pageRankCudaLoopStreamed(G, B, s1, s2, s3, e, r0, eD, r0D, aD, fD, rD, cD, vfromD, efromD, vdataD, N, 0, M, p, E, S);
}
*/
