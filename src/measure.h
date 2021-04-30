#pragma once
#include <chrono>
#include "_cuda.h"

using std::chrono::microseconds;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;




// In milliseconds.
template <class F>
float measureDuration(F fn, int n=_REPEAT) {
  auto start = high_resolution_clock::now();

  for (int i=0; i<n; i++)
    fn();

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  return duration.count()/(n*1000.0f);
}




template <class F>
float measureDurationCuda(F fn, int n=_REPEAT) {
  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );
  TRY( cudaEventRecord(start, 0) );

  for (int i=0; i<n; i++)
    fn();

  float duration;
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  return duration/n;
}
