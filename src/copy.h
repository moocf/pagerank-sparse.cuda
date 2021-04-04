#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include "_cuda.h"

using std::vector;
using std::unordered_map;
using std::fill;
using std::max;




// COPY
// ----

template <class T>
void copy(T *a, T *x, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i];
}

template <class T>
void copy(vector<T>& a, vector<T>& x) {
  copy(a.data(), x.data(), x.size());
}

template <class K, class T>
void copy(unordered_map<K, T>& a, unordered_map<K, T>& x) {
  for (auto& p : x)
    a[p.first] = p.second;
}




// COPY-ABS
// --------

template <class T>
void copyAbs(T *a, T *x, int N) {
  for (int i=0; i<N; i++)
    a[i] = abs(x[i]);
}

template <class T>
void copyAbs(vector<T>& a, vector<T>& x) {
  copyAbs(a.data(), x.data(), x.size());
}

template <class K, class T>
void copyAbs(unordered_map<K, T>& a, unordered_map<K, T>& x) {
  for (auto& p : x)
    a[p.first] = abs(p.second);
}




// COPY-AT
// -------

template <class T, class I>
void copyAt(T *a, T *x, I&& is) {
  for (int i : is)
    a[i] = x[i];
}

template <class T, class I>
void copyAt(vector<T>& a, vector<T>& x, I&& is) {
  copyAt(a.data(), x.data(), is);
}

template <class K, class T, class I>
void copyAt(unordered_map<K, T>& a, unordered_map<K, T>& x, I&& ks) {
  for (auto&& k : ks)
    a[k] = x[k];
}




// COPY-ABS-AT
// -----------

template <class T, class I>
void copyAbsAt(T *a, T *x, I&& is) {
  for (int i : is)
    a[i] = abs(x[i]);
}

template <class T, class I>
void copyAbsAt(vector<T>& a, vector<T>& x, I&& is) {
  copyAbsAt(a.data(), x.data(), is);
}

template <class K, class T, class I>
void copyAbsAt(unordered_map<K, T>& a, unordered_map<K, T>& x, I&& ks) {
  for (auto&& k : ks)
    a[k] = abs(x[k]);
}




// COPY (CUDA)
// -----------

template <class T>
__device__ void copyKernelLoop(T *a, T *x, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = x[i];
}


template <class T>
__global__ void copyKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);

  copyKernelLoop(a, x, N, B*b+t, G*B);
}




// COPY-ABS (CUDA)
// ---------------

template <class T>
__device__ void copyAbsKernelLoop(T *a, T *x, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = abs(x[i]);
}


template <class T>
__global__ void copyAbsKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);

  copyAbsKernelLoop(a, x, N, B*b+t, G*B);
}




// COPY-AT (CUDA)
// --------------

template <class T>
__device__ void copyAtKernelLoop(T *a, T *x, int *is, int IS, int i, int DI) {
  for (; i<IS; i+=DI)
    a[is[i]] = x[is[i]];
}


template <class T>
__global__ void copyAtKernel(T *a, T *x, int *is, int IS) {
  DEFINE(t, b, B, G);

  copyAtKernelLoop(a, x, is, IS, B*b+t, G*B);
}




// COPY-ABS-AT (CUDA)
// ------------------

template <class T>
__device__ void copyAbsAtKernelLoop(T *a, T *x, int *is, int IS, int i, int DI) {
  for (; i<IS; i+=DI)
    a[is[i]] = abs(x[is[i]]);
}


template <class T>
__global__ void copyAbsAtKernel(T *a, T *x, int *is, int IS) {
  DEFINE(t, b, B, G);

  copyAbsAtKernelLoop(a, x, is, IS, B*b+t, G*B);
}




// COPY (GRAPH)
// ------------

template <class G, class H>
void copy(G& a, H& x) {
  for (auto u : x.vertices())
    a.addVertex(u, x.vertexData(u));
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a.addEdge(u, v, x.edgeData(u, v));
  }
}

template <class G>
auto copy(G& x) {
  G a; copy(a, x);
  return a;
}
