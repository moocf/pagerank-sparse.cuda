#pragma once
#include "src/main.h"

using namespace std;




void runAdd() {
  int N = 64*1024*1024;
  float *x = new float[N], t;
  vector<int> is1(N);
  for (int i=0; i<N; i++) is1[i] = i;
  auto is2 = range(N);
  auto is3 = filter(is2, [&](int i) { return true; });
  t = measureDuration([&]() { add(x, N, 1.0f); });
  printf("[%07.1f ms] add(x, N, v)\n", t);
  t = measureDuration([&]() { addAt(x, is1, 1.0f); });
  printf("[%07.1f ms] addAt(x, <vector> is, v)\n", t);
  t = measureDuration([&]() { addAt(x, is2, 1.0f); });
  printf("[%07.1f ms] addAt(x, <range>  is, v)\n", t);
  t = measureDuration([&]() { addAt(x, is3, 1.0f); });
  printf("[%07.1f ms] addAt(x, <filter> is, v)\n", t);
}


void runFill() {
  int N = 64*1024*1024;
  float *x = new float[N], t;
  t = measureDuration([&]() { fill(x, N, 1.0f); });
  printf("[%07.1f ms] fill\n", t);
  t = measureDuration([&]() { fillOmp(x, N, 1.0f); });
  printf("[%07.1f ms] fillOmp\n", t);
  t = measureDuration([&]() { fillCuda(x, N, 1.0f); });
  printf("[%07.1f ms] fillCuda\n", t);
  delete[] x;
}


void runSum() {
  int N = 64*1024*1024;
  float *x = new float[N], t;
  fill(x, N, 1.0f);
  t = measureDuration([&]() { sum(x, N); });
  printf("[%07.1f ms] sum\n", t);
  t = measureDuration([&]() { sumOmp(x, N); });
  printf("[%07.1f ms] sumOmp\n", t);
  t = measureDuration([&]() { sumCuda(x, N); });
  printf("[%07.1f ms] sumCuda\n", t);
  delete[] x;
}


void runErrorAbs() {
  int N = 64*1024*1024;
  float *x = new float[N];
  float *y = new float[N], t;
  fill(x, N, 1.0f);
  fill(y, N, 2.0f);
  t = measureDuration([&]() { errorAbs(x, y, N); });
  printf("[%07.1f ms] errorAbs\n", t);
  t = measureDuration([&]() { errorAbsOmp(x, y, N); });
  printf("[%07.1f ms] errorAbsOmp\n", t);
  t = measureDuration([&]() { errorAbsCuda(x, y, N); });
  printf("[%07.1f ms] errorAbsCuda\n", t);
  delete[] x;
  delete[] y;
}


void runDotProduct() {
  int N = 64*1024*1024;
  float *x = new float[N];
  float *y = new float[N], t;
  fill(x, N, 1.0f);
  fill(y, N, 1.0f);
  t = measureDuration([&]() { dotProduct(x, y, N); });
  printf("[%07.1f ms] dotProduct\n", t);
  t = measureDuration([&]() { dotProductOmp(x, y, N); });
  printf("[%07.1f ms] dotProductOmp\n", t);
  t = measureDuration([&]() { dotProductCuda(x, y, N); });
  printf("[%07.1f ms] dotProductCuda\n", t);
  delete[] x;
  delete[] y;
}
