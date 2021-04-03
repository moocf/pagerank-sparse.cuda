#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include "_cuda.h"

using std::vector;
using std::unordered_map;
using std::max;




// Adds a value to all elements of an array.
// \param a array
// \param N size of array
// \param v value to add
template <class T>
void add(T *a, int N, T v) {
  for (int i=0; i<N; i++)
    a[i] += v;
}


// Adds a value to all elements of a container.
// \param a container (vector, array)
// \param v value to add
template <class C, class T>
void add(C& a, T v) {
  add(a.data(), a.size(), v);
}


// Adds a value to all elements of a map.
// \param a map (ordered, unordered)
// \param v value to add
template <class K, class T>
void add(unordered_map<K, T>& a, T v) {
  for (auto&& p : a)
    p.second += v;
}




// Adds a value to certain indices of an array.
// \param a array
// \param is indices
// \param v value to add
template <class I, class T>
void addAt(T *a, I&& is , T v) {
  for (int i : is)
    a[i] += v;
}


// Adds a value to certain indices of a container.
// \param a container
// \param is indices
// \param v value to add
template <class C, class I, class T>
void addAt(C& a, I&& is, T v) {
  addAt(a.data(), is, v);
}


// Adds a value to certain keys of a map.
// \param a map (ordered, unordered)
// \param ks keys
// \param v value to add
template <class K, class I, class T>
void addAt(unordered_map<K, T>& a, I&& ks, T v) {
  for (auto&& k : ks)
    a[k] += v;
}




// Adds values of two arrays respectively.
// \param a answer array
// \param x first array
// \param y second array
// \param N size of arrays
template <class T>
void add(T *a, T *x, T *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i] + y[i];
}


// Adds values of two containers respectively.
// \param a answer container (vector, array)
// \param x first container
// \param y second container
template <class C, class T>
void add(C& a, C&& x, C&& y) {
  return add(a.data(), x.data(), y.data(), x.size());
}


// Adds values of two maps respectively.
// \param a answer map (ordered, unordered)
// \param x first map
// \param y second map
template <class K, class T>
void add(unordered_map<K, T>& a, unordered_map<K, T>&& x, unordered_map<K, T>&& y) {
  for (auto&& p : x)
    a[p.first] = x[p.first] + y[p.first];
}




// Adds a value to all elements of an array.
// \param a array
// \param N size of array
// \param v value to add
template <class T>
void addOmp(T *a, int N, T v) {
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    a[i] += v;
}


// Adds a value to all elements of an array.
// \param a array
// \param N size of array
// \param v value to add
template <class C, class T>
void addOmp(C& a, T v) {
  addOmp(a.data(), a.size(), v);
}




// Adds a value to all elements of an array in steps (1 thread).
// \param a array
// \param N size of array
// \param v value to add
// \param i start index
// \param DI step size
template <class T>
__device__ void addKernelLoop(T *a, int N, T v, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] += v;
}

// Adds a value to all elements of an array.
// \param a array
// \param N size of array
// \param v value to add
template <class T>
__global__ void addKernel(T *a, int N, T v) {
  DEFINE(t, b, B, G);

  addKernelLoop(a, N, v, B*b+t, G*B);
}


// Adds a value to all elements of an array.
// \param a array
// \param N size of array
// \param v value to add
template <class T>
void addCuda(T *a, int N, T v) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *aD;
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMemcpy(aD, a, N1, cudaMemcpyHostToDevice) );

  addKernel<<<G, B>>>(aD, N, v);
  TRY( cudaMemcpy(a, aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(aD) );
}


// Adds a value to all elements of a container.
// \param a container (vector, array)
// \param v value to add
template <class C, class T>
void addCuda(C& a, T v) {
  addCuda(a.data(), a.size(), v);
}




// Adds values of two arrays respectively in steps (1 thread).
// \param a answer array
// \param x first array
// \param y second array
// \param N size of arrays
// \param i start index
// \param DI step size
template <class T>
__device__ void addKernelLoop(T *a, T *x, T *y, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = x[i] + y[i];
}


// Adds values of two arrays respectively.
// \param a answer array
// \param x first array
// \param y second array
// \param N size of arrays
template <class T>
__global__ void addKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);

  addKernelLoop(a, x, y, N, B*b+t, G*B);
}


// Adds values of two arrays respectively.
// \param a answer array
// \param x first array
// \param y second array
// \param N size of arrays
template <class T>
void addCuda(T *a, T *x, T *y, int N) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  size_t N1 = N * sizeof(T);

  T *xD, *yD;
  TRY( cudaMalloc(&xD, N1) );
  TRY( cudaMalloc(&yD, N1) );
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) );

  addKernel<<<G, B>>>(xD, xD, yD, N);
  TRY( cudaMemcpy(a, xD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFree(xD) );
  TRY( cudaFree(yD) );
}


// Adds values of two containers respectively.
// \param a answer container
// \param x first container
// \param y second container
template <class C, class T>
void addCuda(C& a, C& x, C& y) {
  addCuda(a.data(), x.data(), y.data(), a.size());
}
