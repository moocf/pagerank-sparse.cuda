#pragma once
#include <chrono>
#include "_cuda.h"

using std::chrono::milliseconds;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;



// In milliseconds.
template <class F>
float measureDuration(F fn) {
  auto start = high_resolution_clock::now();

  fn();

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  return duration.count();
}




template <class F>
float measureDurationCuda(F fn) {
  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );
  TRY( cudaEventRecord(start, 0) );

  fn();

  float duration;
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  return duration;
}
