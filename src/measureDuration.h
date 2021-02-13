#pragma once
#include <time.h>
#include "_cuda.h"




// In milliseconds.
template <class F>
float measureDuration(F fn) {
  clock_t start = clock();

  fn();

  clock_t stop = clock();
  float duration = (float) (stop - start) / CLOCKS_PER_SEC;
  return duration * 1000;
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
