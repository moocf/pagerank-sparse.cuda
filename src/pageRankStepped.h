#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>
#include "_cuda.h"
#include "ceilDiv.h"
#include "measure.h"
#include "sum.h"
#include "fill.h"
#include "error.h"
#include "join.h"
#include "vertices.h"
#include "edges.h"
#include "components.h"
#include "blockgraph.h"
#include "sort.h"
#include "crossEdges.h"
#include "difference.h"
#include "pageRank.h"




// PAGE-RANK STEPPED HELPERS (CUDA)
// --------------------------------

template <class G, class K>
auto pageRankWaves(G& xt, vector<vector<K>>& cs, PageRankMode M) {
  vector<vector<int>> a;
  for (auto& c : cs)
    a.push_back(pageRankStep(xt, c, M));
  return a;
}




// PAGE-RANK STEPPED (CUDA)

template <class T, class I>
T* pageRankSteppedCudaOnce(float& R, T *e, T *r0, T *eD, T *r0D, T *aD, T *cD, T *rD, T *fD, T *qD, int *vfromD, int *efromD, int *vcrosD, int *ecrosD, int *vdataD, int *vrootD, int *vdistD, int i, I&& ls, int n, int N, T p, T E, bool fSC) {
  for (auto ns : ls) {
    int n = sumAbs(ns);
    if (qD) pageRankKernelWave(qD, (T*) nullptr, rD, cD, vcrosD, ecrosD, T(), false, i, ns);
    T *bD = pageRankCudaLoop(R, e, r0, eD, r0D, aD, cD, rD, fD, qD, vfromD, efromD, vdataD, vrootD, vdistD, i, ns, n, N, p, E, fSC);
    if (bD != rD) swap(aD, rD);
    i += n;
  }
  return rD;
}


template <class T, class I>
T* pageRankSteppedCudaLoop(T *e, T *r0, T *eD, T *r0D, T *aD, T *cD, T *rD, T *fD, int *vfromD, int *efromD, int *vdataD, int *vrootD, int *vdistD, int i, I&& ls, int n, int N, T p, T E, bool fSC, int d0) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  int G1 = G * sizeof(T);
  T e0 = 0;
  while (1) {
    T *bD = pageRankSticdCudaStep(e, r0, eD, r0D, aD, cD, rD, fD, vfromD, efromD, vdataD, i, ls, n, N, p, E, fSC);
    if (bD != rD) swap(aD, rD);
    if (d0 == 0) break;
    absErrorKernel<<<G, B>>>(eD, rD, aD, N);
    TRY( cudaMemcpy(e, eD, G1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, G);
    if (e1 < E || e1 == e0) break;
    e0 = e1;
  }
  return rD;
}


template <class T, class I>
T* pageRankSteppedCudaCore(float& R, T *e, T *r0, T *eD, T *r0D, T *aD, T *cD, T *rD, T *fD, T *qD, int *vfromD, int *efromD, int *vcrosD, int *ecrosD, int *vdataD, int *vrootD, int *vdistD, I&& ls, int N, T p, T E, bool fSC) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  fillKernel<<<G, B>>>(rD, N, T(1)/N);
  if (qD) fillKernel<<<G, B>>>(qD, N, T());
  pageRankFactorKernel<<<G, B>>>(fD, vdataD, p, N);
  return pageRankSteppedCudaOnce(R, e, r0, eD, r0D, aD, cD, rD, fD, qD, vfromD, efromD, vcrosD, ecrosD, vdataD, vrootD, vdistD, 0, ls, N, N, p, E, fSC);
}


template <class G, class H, class C, class T=float>
auto pageRankSteppedCuda(float& t, float& R, G& x, H& xt, H& xe, H& xf, C& xcs, C& xid, C& xch, PageRankOptions<T> o=PageRankOptions<T>()) {
  using K = typename G::TKey;
  vector<vector<K>> ks0;
  auto M = o.mode;
  auto F = o.flags;
  auto p = o.damping;
  auto E = o.convergence;
  F.splitComponents = true;
  F.orderComponents = true;
  bool fCP = F.crossPropagate;
  bool fRI = F.removeIdenticals;
  bool fRC = F.removeChains;
  bool fSC = F.skipConverged;
  auto& ch = fRC? xch : ks0;
  auto& id = fRI? xid : ks0;
  auto cs = pageRankComponents(x, xt, xcs, ch, id, M, F);
  auto ls = pageRankWaves(xt, cs, M);
  auto ks = join(cs);
  auto vfrom = sourceOffsets(fCP? xf : xt, ks);
  auto efrom = destinationIndices(fCP? xf : xt, ks);
  auto vcros = fCP? sourceOffsets(xe, ks)      : vector<int>();
  auto ecros = fCP? destinationIndices(xe, ks) : vector<int>();
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
  int VCROS1 = vcros.size() * sizeof(int);
  int ECROS1 = ecros.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int VROOT1 = vroot.size() * sizeof(int);
  int VDIST1 = vdist.size() * sizeof(int);
  int G1 = g * sizeof(T);
  int N1 = N * sizeof(T);
  vector<T> a(N);

  T *e,  *r0,  *qD = NULL;
  T *eD, *r0D, *fD, *rD, *cD, *aD, *bD;
  int *vfromD = NULL, *efromD = NULL;
  int *vcrosD = NULL, *ecrosD = NULL;
  int *vdataD = NULL, *vrootD = NULL, *vdistD = NULL;
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
  if (fCP) TRY( cudaMalloc(&qD, N1) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  if (fCP) TRY( cudaMalloc(&vcrosD, VCROS1) );
  if (fCP) TRY( cudaMalloc(&ecrosD, ECROS1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  if (fRI || fRC) TRY( cudaMalloc(&vrootD, VROOT1) );
  if (fRI || fRC) TRY( cudaMalloc(&vdistD, VDIST1) );
  TRY( cudaMemcpyAsync(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaMemcpyAsync(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaMemcpyAsync(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice, s1) );
  if (fRI || fRC) TRY( cudaMemcpyAsync(vrootD, vroot.data(), VROOT1, cudaMemcpyHostToDevice, s1) );
  if (fRI || fRC) TRY( cudaMemcpyAsync(vdistD, vdist.data(), VDIST1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaStreamSynchronize(s1) );

  t = measureDuration([&]() { bD = pageRankSteppedCudaCore(R, e, r0, eD, r0D, aD, cD, rD, fD, qD, vfromD, efromD, vcrosD, ecrosD, vdataD, vrootD, vdistD, ls, N, p, E, fSC); });
  TRY( cudaMemcpy(a.data(), bD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(eD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(aD) );
  if (fCP) TRY( cudaFree(qD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  if (fCP) TRY( cudaFree(vcrosD) );
  if (fCP) TRY( cudaFree(ecrosD) );
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
auto pageRankSteppedCuda(float& t, float& R, G& x, H& xt, PageRankOptions<T> o=PageRankOptions<T>()) {
  int GB = GRID_DIM * BLOCK_DIM;
  auto cs = components(x, xt);
  auto ds = joinUntilSize(cs, GB);
  auto xe = crossEdges(xt, ds);
  auto xf = edgeDifference(xt, xe);
  auto id = inIdenticals(x, xt);
  auto ch = chains(x, xt);
  return pageRankSteppedCuda(t, R, x, xt, cs, id, ch, o);
}
