#pragma once
#include "DiGraphView.h"
#include "join.h"
#include "transform.h"
#include "transpose.h"
#include "components.h"
#include "blockgraph.h"
#include "sort.h"
#include "subgraph.h"
#include "pageRank.h"




template <class G, class C, class T>
auto& pageRankSticdLoop(C& a, C& r, C& f, C& c, G& xs, T p, T E) {
  for (auto& x : xs)
    pageRankCore(a, r, f, c, x, p, E);
  return a;
}

template <class G, class C, class T>
auto& pageRankSticdCore(C& a, C& r, C& f, C& c, G& x, G& xs, T p, T E) {
  int N = x.order();
  fillAt(r, x.vertices(), T(1)/N);
  pageRankFactor(f, xs, p);
  return pageRankSticdLoop(a, r, f, c, xs, p, E);
}


template <class G, class H, class T>
auto pageRankSticd(float& t, G& x, H& y, T p, T E) {
  // using G = decltype(x.vertexContainer(T()));
  auto cs  = components(x, y);
  auto b   = blockgraph(x, cs);
  auto bks = sort(b);
  vector<DiGraphView<H>> ys;
  // for (int i : bks)
  //   ys.push_back(subgraph(y, cs[i]));
  auto a = x.vertexContainer(T());
  // auto r = x.vertexContainer(T());
  // auto f = x.vertexContainer(T());
  // auto c = x.vertexContainer(T());
  // t = measureDuration([&]() { pageRankSticdCore(a, r, f, c, y, ys, p, E); });
  return a;
}

template <class G, class H, class T=float>
auto pageRankSticd(float& t, G& x, H& y, PageRankOptions<T> o=PageRankOptions<T>()) {
  return pageRankSticd(t, x, y, o.damping, o.convergence);
}



template <class T>
T* pageRankSticdCudaStep(int G, int B, T *e, T *r0, T *eD, T *r0D, T *aD, T *fD, T *rD, T *cD, int *vfromD, int *efromD, int *vdataD, int N, vector<int>& CS, PageRankMode M, T p, T E, int S) {
  int i = 0;
  for (auto C : CS) {
    T *bD = pageRankCudaLoop(G, B, e, r0, eD, r0D, aD, fD, rD, cD, vfromD, efromD, vdataD, N, i, C, M, p, E, S);
    if (bD != rD) swap(aD, rD);
    i += C;
  }
  return rD;
}

template <class T>
T* pageRankSticdCudaLoop(int G, int B, T *e, T *r0, T *eD, T *r0D, T *aD, T *fD, T *rD, T *cD, int *vfromD, int *efromD, int *vdataD, int N, vector<int>& CS, PageRankMode M, T p, T E, int S) {
  // T e0 = 0;
  int G1 = G * sizeof(T);
  for (int z=0; z<25; z++) {
    T *bD = pageRankSticdCudaStep(G, B, e, r0, eD, r0D, aD, fD, rD, cD, vfromD, efromD, vdataD, N, CS, M, p, E, S);
    if (bD != rD) swap(aD, rD);
    errorAbsKernel<<<G, B>>>(eD, rD, aD, N);
    TRY( cudaMemcpy(e, eD, G1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, G);
    printf("e1: %.23f, E: %.23f\n", e1, E);
    // if (e1 < E || e1 == e0) break;
    // e0 = e1;
  }
  return rD;
}

template <class T>
T* pageRankSticdCudaCore(T *e, T *r0, T *eD, T *r0D, T *aD, T *fD, T *rD, T *cD, int *vfromD, int *efromD, int *vdataD, int N, vector<int>& CS, PageRankMode M, T p, T E, int S) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  pageRankFactorKernel<<<G, B>>>(fD, vdataD, p, N);
  fillKernel<<<G, B>>>(rD, N, T(1)/N);
  return pageRankSticdCudaLoop(G, B, e, r0, eD, r0D, aD, fD, rD, cD, vfromD, efromD, vdataD, N, CS, M, p, E, S);
}




template <class G, class H, class T>
auto pageRankSticdCuda(float& t, G& x, H& y, PageRankMode M, T p, T E) {
  using K = typename G::TKey;
  auto cs  = components(x, y);
  auto b   = blockgraph(x, cs);
  auto bks = sort(b);
  auto ks  = joinFrom(cs, bks);
  vector<int> CS  = transformFrom(cs, bks, [](auto& c) { return (int) c.size(); });
  auto vfrom = sourceOffsets(y, ks);
  auto efrom = destinationIndices(y, ks);
  auto vdata = vertexData(y, ks);  // outDegree
  // printf("x: "); print(x, true);
  // printf("y: "); print(y, true);
  // printf("cs: \n");
  // for (auto& c : cs)
  //   print(c);
  // printf("b: "); print(b, true);
  // printf("bks: "); print(bks);
  // printf("ks: "); print(ks);
  // printf("CS: "); print(CS);
  // printf("vdata: "); print(vdata);
  // printf("vfrom: "); print(vfrom);
  int N = x.order(), S = 0;
  int B = BLOCK_DIM;
  int g = min(ceilDiv(N, B), GRID_DIM);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int G1 = g * sizeof(T);
  int N1 = N * sizeof(T);
  vector<T> a(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD, *bD;
  int *vfromD, *efromD, *vdataD;
  cudaStream_t s1, s2, s3;
  TRY( cudaProfilerStart() );
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
  TRY( cudaMemcpyAsync(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaMemcpyAsync(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaMemcpyAsync(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice, s1) );
  TRY( cudaStreamSynchronize(s1) );

  t = measureDuration([&]() { bD = pageRankSticdCudaCore(e, r0, eD, r0D, aD, fD, rD, cD, vfromD, efromD, vdataD, N, CS, M, p, E, S); });
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
  TRY( cudaStreamDestroy(s1) );
  TRY( cudaStreamDestroy(s2) );
  TRY( cudaStreamDestroy(s3) );
  TRY( cudaProfilerStop() );
  return vertexContainer(x, a, ks);
}

template <class G, class H, class T=float>
auto pageRankSticdCuda(float& t, G& x, H& y, PageRankOptions<T> o=PageRankOptions<T>()) {
  return pageRankSticdCuda(t, x, y, o.mode, o.damping, o.convergence);
}
