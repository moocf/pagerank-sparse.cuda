#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include "_support.h"




#ifndef _THREADS
/** Default threads per block. */
#define _THREADS 64
#endif

#ifndef _BLOCKS
/** Max. blocks per grid. */
#define _BLOCKS 1024
#endif




#ifndef TRY_CUDA
void try_cuda(cudaError err, const char* exp, const char* func, int line, const char* file) {
  if (err == cudaSuccess) return;
  fprintf(stderr,
      "%s: %s\n"
      "  in expression %s\n"
      "  at %s:%d in %s\n",
      cudaGetErrorName(err), cudaGetErrorString(err),
      exp,
      func, line, file);
  exit(err);
}

// Prints an error message and exits, if CUDA expression fails.
// TRY_CUDA( cudaDeviceSynchronize() );
#define TRY_CUDA(exp) try_cuda(exp, #exp, __func__, __LINE__, __FILE__)
#endif

#ifndef TRY
#define TRY(exp) TRY_CUDA(exp)
#endif




#ifndef DEFINE_CUDA
// Defines short names for the following variables:
// - threadIdx.x,y
// - blockIdx.x,y
// - blockDim.x,y
// - gridDim.x,y
#define DEFINE_CUDA(t, b, B, G) \
  int t = threadIdx.x; \
  int b = blockIdx.x; \
  int B = blockDim.x; \
  int G = gridDim.x;
#define DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY) \
  int tx = threadIdx.x; \
  int ty = threadIdx.y; \
  int bx = blockIdx.x; \
  int by = blockIdx.y; \
  int BX = blockDim.x; \
  int BY = blockDim.y; \
  int GX = gridDim.x;  \
  int GY = gridDim.y;
#endif

#ifndef DEFINE
#define DEFINE(t, b, B, G) \
  DEFINE_CUDA(t, b, B, G)
#define DEFINE2D(tx, ty, bx, by, BX, BY, GX, GY) \
  DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY)
#endif




template <class T>
__device__ void unusedCuda(T&&) {}

#ifndef UNUSED_CUDA
#define UNUSED_CUDA(...) ARG_CALL(unusedCuda, ##__VA_ARGS__)
#endif

#ifndef UNUSED
#define UNUSED UNUSED_CUDA
#endif




#ifndef __SYNCTHREADS
void __syncthreads();
#define __SYNCTHREADS() __syncthreads()
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __shared__
#define __shared__
#endif
