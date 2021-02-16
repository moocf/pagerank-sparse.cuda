#include <array>
#include <stdio.h>
#include "src/main.h"

using namespace std;




const char* testRange() {
  int n = 0;
  double v = 0, V = 10, DV = 0.5;
  for (double x : rangeIterable(v, V, DV))
    if (x != v+DV*(n++)) return "rangeIterable";
  if (n != 20) return "rangeIterable";
  return NULL;
}


const char* testTransform() {
  // int n = 0;
  // double v = 0, V = 10, DV = 0.5;
  // for (double x : transform(rangeIterable(v, V, DV), [](double a) { return a*2; })) {
  //   if ((int) x != x) return "transform";
  //   n++;
  // }
  // if (n != 20) return "transform";
  return NULL;
}


const char* testFill() {
  array<int, 4> x = {1, 2, 3, 4};
  array<int, 4> a;

  a = x;
  fill(a, 4);
  for (auto& v : a)
    if (v != 4) return "fill";

  a = x;
  fillOmp(a, 4);
  for (auto& v : a)
    if (v != 4) return "fillOmp";

  a = x;
  fillCuda(a, 4);
  for (auto& v : a)
    if (v != 4) return "fillCuda";
  return NULL;
}


const char* testSum() {
  array<int, 4> x = {1, 2, 3, 4};
  int a;

  a = sum(x);
  if (a != 10) return "sum";

  a = sumOmp(x);
  if (a != 10) return "sumOmp";

  a = sumCuda(x);
  if (a != 10) return "sumCuda";
  return NULL;
}


const char* testDotProduct() {
  array<int, 4> x = {1, 2, 3, 4};
  array<int, 4> y = {1, 0, 1, 0};
  int a;

  a = dotProduct(x, y);
  if (a != 4) return "dotProduct";

  a = dotProductCuda(x, y);
  if (a != 4) return "dotProductCuda";
  return NULL;
}


const char* testErrorAbs() {
  array<int, 4> x = {1, 2, 3, 4};
  array<int, 4> y = {1, 1, 3, 5};
  int a;

  a = errorAbs(x, y);
  if (a != 2) return "errorAbs";

  a = errorAbsCuda(x, y);
  if (a != 2) return "errorAbsCuda";
  return NULL;
}


const char* testAll() {
  vector<const char*> ts = {
    testRange(),
    testTransform(),
    testFill(),
    testSum(),
    testDotProduct(),
    testErrorAbs()
  };
  const char *e = NULL;
  for (auto& t : ts) {
    if (t) printf("ERROR: %s() failed!\n", t);
    e = e? e : t;
  }
  return e;
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


void runPageRank(DiGraph& g) {
  float t;
  vector<float> ranks;
  t = measureDuration([&]() { ranks = pageRank(g); });
  printf("[%07.1f ms] pageRank     \n", t); print(ranks);
  // t = measureDuration([&]() { pageRankOmp(ranks, g); });
  // printf("[%07.1f ms] pageRankOmp  = \n", t); // print(ranks, N);
  // t = measureDuration([&]() { pageRankCuda(ranks, g); });
  // printf("[%07.1f ms] pageRankCuda = \n", t); // print(ranks, N);
}




int main(int argc, char **argv) {
  printf("Loading graph ...\n");
  DiGraph g;
  readMtx(g, argv[1]);
  print(g);
  testAll();
  runGraph();
  runFill();
  runSum();
  runErrorAbs();
  runDotProduct();
  // runPageRank(g);
  return 0;
}
