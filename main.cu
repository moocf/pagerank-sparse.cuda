#include <array>
#include <vector>
#include <stdio.h>
#include "src/main.h"

using namespace std;




const char* testCeilDiv() {
  if (!(
    ceilDiv(0, 5) == 0 &&
    ceilDiv(1, 5) == 1 &&
    ceilDiv(4, 5) == 1 &&
    ceilDiv(5, 5) == 1 &&
    ceilDiv(6, 5) == 2
  )) return "ceilDiv(int)";
  if (!(
    ceilDiv(0.0f, 0.5f) == 0.0f &&
    ceilDiv(0.1f, 0.5f) == 1.0f &&
    ceilDiv(0.4f, 0.5f) == 1.0f &&
    ceilDiv(0.5f, 0.5f) == 1.0f &&
    ceilDiv(0.6f, 0.5f) == 2.0f
  )) return "ceilDiv(float)";
  if (!(
    ceilDiv(0.0, 0.5) == 0.0 &&
    ceilDiv(0.1, 0.5) == 1.0 &&
    ceilDiv(0.4, 0.5) == 1.0 &&
    ceilDiv(0.5, 0.5) == 1.0 &&
    ceilDiv(0.6, 0.5) == 2.0
  )) return "ceilDiv(double)";
  return NULL;
}


const char* testErase() {
  vector<int> a {0, 1, 2, 3};
  vector<int> b {0, 1, 2, 3};
  vector<int> c {0, 1, 2, 3};
  vector<int> d {0, 1, 2, 3};
  vector<int> e {0, 1, 2, 3};
  vector<int> f {0, 1, 2, 3};
  eraseAt(a, 0);    // 123
  eraseAt(b, 0, 2); // 23
  eraseAt(c, 1);    // 023
  eraseAt(d, 1, 2); // 03
  eraseAt(e, 3);    // 012
  eraseAt(f, 2, 2); // 01
  if (a[0] != 1 || a.size() != 3) return "eraseAtFront1";
  if (b[0] != 2 || b.size() != 2) return "eraseAtFront2";
  if (c[1] != 2 || c.size() != 3) return "eraseAtMid1";
  if (d[1] != 3 || d.size() != 2) return "eraseAtMid2";
  if (e[2] != 2 || e.size() != 3) return "eraseAtBack1";
  if (f[1] != 1 || f.size() != 2) return "eraseAtBack2";
  return NULL;
}


const char* testInsert() {
  vector<int> a {0, 1, 2, 3};
  vector<int> b {0, 1, 2, 3};
  vector<int> c {0, 1, 2, 3};
  vector<int> d {0, 1, 2, 3};
  vector<int> e {0, 1, 2, 3};
  vector<int> f {0, 1, 2, 3};
  insertAt(a, 0, 9);    // 90123
  insertAt(b, 0, 2, 9); // 990123
  insertAt(c, 1, 9);    // 09123
  insertAt(d, 1, 2, 9); // 099123
  insertAt(e, 4, 9);    // 01239
  insertAt(f, 4, 2, 9); // 012399
  if (a[0] != 9 || a.size() != 5) return "insertAtFront1";
  if (b[1] != 9 || b.size() != 6) return "insertAtFront2";
  if (c[1] != 9 || c.size() != 5) return "insertAtMid1";
  if (d[2] != 9 || d.size() != 6) return "insertAtMid2";
  if (e[4] != 9 || e.size() != 5) return "insertAtBack1";
  if (f[5] != 9 || f.size() != 6) return "insertAtBack2";
  return NULL;
}


const char* testFilter() {
  vector<int> a {0, 1, 2, 3, 4, 5, 6, 7};
  auto b = filter(a.begin(), a.end(), [](int v) { return v % 4 != 0; });
  auto c = filter(b.begin(), b.end(), [](int v) { return v % 3 != 0; });
  auto d = filter(c.begin(), c.end(), [](int v) { return v % 2 != 0; });
  auto e = filter(a, [](int v) { return v % 4 != 0; }); // 123567
  auto f = filter(e, [](int v) { return v % 3 != 0; }); // 1257
  auto g = filter(f, [](int v) { return v % 2 != 0; }); // 157
  if (b.size() != 6) return "filterIBIE1";
  if (c.size() != 4) return "filterIBIE2";
  if (d.size() != 3) return "filterIBIE3";
  if (e.size() != 6) return "filterC1";
  if (f.size() != 4) return "filterC2";
  if (g.size() != 3) return "filterC3";
  return NULL;
}


const char* testLowerBound() {
  vector<int> a {0, 1, 2, 3};
  auto i = lowerBound(a.begin(), a.end(), 2)-a.begin();
  auto j = lowerBound(a, 2)-a.begin();
  auto k = lowerBoundAt(a, 2);
  if (i != 2) return "lowerBoundIBIE";
  if (j != 2) return "lowerBoundC";
  if (k != 2) return "lowerBoundAt";
  return NULL;
}


const char* testScan() {
  vector<int> a {0, 1, 2, 3};
  auto i = scan(a.begin(), a.end(), 2)-a.begin();
  auto j = scan(a.begin(), a.end(), 9)-a.begin();
  auto k = scan(a, 2)-a.begin();
  auto l = scan(a, 9)-a.begin();
  auto m = scanAt(a, 2);
  auto n = scanAt(a, 9);
  if (i != 2) return "scanIBIE1";
  if (j != 4) return "scanIBIE2";
  if (k != 2) return "scanC1";
  if (l != 4) return "scanC2";
  if (m != 2) return "scanAt1";
  if (n != 4) return "scanAt2";
  return NULL;
}


const char* testTransform() {
  vector<int> a {0, 1, 2, 3};
  auto b = transform(a.begin(), a.end(), [](int v) { return v+1; });
  auto c = transform(b.begin(), b.end(), [](int v) { return v+1; });
  auto d = transform(c.begin(), c.end(), [](int v) { return v+1; });
  auto e = transform(a, [](int v) { return v+1; }); // 1234
  auto f = transform(e, [](int v) { return v+1; }); // 2345
  auto g = transform(f, [](int v) { return v+1; }); // 3456
  for (auto&& v : b) if (v<1 || v>4) return "transformIBIE1";
  for (auto&& v : c) if (v<2 || v>5) return "transformIBIE2";
  for (auto&& v : d) if (v<3 || v>6) return "transformIBIE3";
  for (auto&& v : e) if (v<1 || v>4) return "transformC1";
  for (auto&& v : f) if (v<2 || v>5) return "transformC2";
  for (auto&& v : g) if (v<3 || v>6) return "transformC3";
  return NULL;
}


const char* testRange() {
  int n = 0;
  double v = 0, V = 10, DV = 0.5;
  for (double x : range(v, V, DV))
    if (x != v+DV*(n++)) return "range";
  if (n != 20) return "range";
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


void testAll() {
  vector<const char*> ts = {
    testCeilDiv(),
    testErase(),
    testInsert(),
    testFilter(),
    testLowerBound(),
    testScan(),
    testRange(),
    testTransform(),
    testFill(),
    testSum(),
    testDotProduct(),
    testErrorAbs()
  };
  int n = 0;
  for (auto& t : ts) {
    if (!t) continue;
    printf("ERROR: %s() failed!\n", t);
    n++;
  }
  printf("%d/%ld tests failed.\n", n, ts.size());
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
  testAll();
  printf("Loading graph ...\n");
  DiGraphTemp<> g;
  readMtx(g, argv[1]);
  print(g);
  runFill();
  runSum();
  runErrorAbs();
  runDotProduct();
  runGraph();
  // runPageRank(g);
  return 0;
}
