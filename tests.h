#pragma once
#include <vector>
#include <unordered_map>
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
  if (distance(b.begin(), b.end()) != 6) return "filterIBIE1";
  if (distance(c.begin(), c.end()) != 4) return "filterIBIE2";
  if (distance(d.begin(), d.end()) != 3) return "filterIBIE3";
  if (distance(e.begin(), e.end()) != 6) return "filterC1";
  if (distance(f.begin(), f.end()) != 4) return "filterC2";
  if (distance(g.begin(), g.end()) != 3) return "filterC3";
  return NULL;
}


const char* testFind() {
  vector<int> a {0, 1, 2, 3};
  auto i = find(a.begin(), a.end(), 2)-a.begin();
  auto j = find(a.begin(), a.end(), 9)-a.begin();
  auto k = find(a, 2)-a.begin();
  auto l = find(a, 9)-a.begin();
  auto m = findAt(a, 2);
  auto n = findAt(a, 9);
  if (i != 2) return "findIBIE1";
  if (j != 4) return "findIBIE2";
  if (k != 2) return "findC1";
  if (l != 4) return "findC2";
  if (m != 2) return "findAt1";
  if (n != -1) return "findAt2";
  return NULL;
}


const char* testLowerBound() {
  vector<int> a {0, 1, 2, 3};
  auto i = lowerBound(a.begin(), a.end(), 2)-a.begin();
  auto j = lowerBound(a.begin(), a.end(), 9)-a.begin();
  auto k = lowerBound(a, 2)-a.begin();
  auto l = lowerBound(a, 9)-a.begin();
  auto m = lowerBoundAt(a, 2);
  auto n = lowerBoundAt(a, 9);
  if (i != 2) return "lowerBoundIBIE1";
  if (j != 4) return "lowerBoundIBIE2";
  if (k != 2) return "lowerBoundC1";
  if (l != 4) return "lowerBoundC2";
  if (m != 2) return "lowerBoundAt1";
  if (n != 4) return "lowerBoundAt2";
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
  vector<int> a {1, 2, 3, 4};
  vector<int> b {1, 2, 3, 4};
  vector<int> c {1, 2, 3, 4};
  vector<int> d {1, 2, 3, 4};
  unordered_map<char, int> m {{'a', 1}, {'b', 2}, {'c', 3}};
  unordered_map<char, int> n {{'a', 1}, {'b', 2}, {'c', 3}};
  vector<int> is {0, 2};
  vector<char> ks {'a', 'c'};

  fill(a, 9);
  fillOmp(b, 9);
  fillCuda(c, 9);
  fill(m, 9);
  fillAt(d, is, 9);
  fillAt(n, ks, 9);
  for (auto& v : a) if (v != 9) return "fillV";
  for (auto& v : b) if (v != 9) return "fillOmpV";
  for (auto& v : c) if (v != 9) return "fillCudaV";
  for (auto& p : m) if (p.second != 9) return "fillM";
  if (!(d[0]==9 && d[1]==2 && d[2]==9 && d[3]==4)) return "fillAtV";
  if (!(n['a']==9 && n['b']==2 && n['c']==9)) return "fillAtM";
  return NULL;
}


const char* testAdd() {
  vector<int> a {1, 2, 3, 4};
  vector<int> b {1, 2, 3, 4};
  vector<int> c {1, 2, 3, 4};
  vector<int> d {1, 2, 3, 4};
  unordered_map<char, int> m {{'a', 1}, {'b', 2}, {'c', 3}};
  unordered_map<char, int> n {{'a', 1}, {'b', 2}, {'c', 3}};
  vector<int> is {0, 2};
  vector<char> ks {'a', 'c'};

  add(a, 1);
  addOmp(b, 1);
  addCuda(c, 1);
  add(m, 1);
  addAt(d, is, 1);
  addAt(n, ks, 1);
  if (!(a[0]==2 && a[1]==3 && a[2]==4 && a[3]==5)) return "addV";
  if (!(b[0]==2 && b[1]==3 && b[2]==4 && b[3]==5)) return "addOmpV";
  if (!(c[0]==2 && c[1]==3 && c[2]==4 && c[3]==5)) return "addCudaV";
  if (!(m['a']==2 && m['b']==3 && m['c']==4)) return "addM";
  if (!(d[0]==2 && d[1]==2 && d[2]==4 && d[3]==4)) return "addAtV";
  if (!(n['a']==2 && n['b']==2 && n['c']==4)) return "addAtM";
  return NULL;
}


const char* testSum() {
  vector<int> a {1, 2, 3, 4};
  unordered_map<char, int> m {{'a', 1}, {'b', 2}, {'c', 3}};
  vector<int> is {0, 2};
  vector<char> ks {'a', 'c'};

  int p = sum(a);
  int q = sumOmp(a);
  int r = sumCuda(a);
  int s = sum(m);
  int t = sumAt(a, is);
  int u = sumAt(m, ks);
  if (p != 10) return "sumV";
  if (q != 10) return "sumOmpV";
  if (r != 10) return "sumCudaV";
  if (s != 6) return "sumM";
  if (t != 4) return "sumAtV";
  if (u != 4) return "sumAtM";
  return NULL;
}


const char* testDotProduct() {
  vector<int> x {1, 2, 3, 4};
  vector<int> y {1, 0, 1, 0};

  int a = dotProduct(x, y);
  int b = dotProductCuda(x, y);
  if (a != 4) return "dotProduct";
  if (b != 4) return "dotProductCuda";
  return NULL;
}


const char* testErrorAbs() {
  vector<int> x {1, 2, 3, 4};
  vector<int> y {1, 1, 3, 5};
  unordered_map<char, int> m {{'a', 1}, {'b', 2}, {'c', 3}};
  unordered_map<char, int> n {{'a', 1}, {'b', 1}, {'c', 4}};

  int a = errorAbs(x, y);
  int b = errorAbsCuda(x, y);
  int c = errorAbs(m, n);
  if (a != 2) return "errorAbsV";
  if (b != 2) return "errorAbsCudaV";
  if (c != 2) return "errorAbsM";
  return NULL;
}


const char* testDiGraph() {
  DiGraph<int, int, int> g;
  DiGraph<char, int, int> h;

  g.addVertex(1, 10);
  g.addEdge(1, 2, 12);
  g.addEdge(2, 4, 24);
  g.addEdge(4, 3, 43);
  g.addEdge(3, 1, 31);
  g.removeEdge(1, 2);

  h.addVertex('a', 10);
  h.addEdge('a', 'b', 12);
  h.addEdge('b', 'd', 24);
  h.addEdge('d', 'c', 43);
  h.addEdge('c', 'a', 31);
  h.removeEdge('a', 'b');

  if (!(
    g.span()  == 5  &&
    g.order() == 4  &&
    g.size()  == 3  &&
    g.hasVertex(1)  && g.hasVertex(2)  && g.hasVertex(3)  && g.hasVertex(4) &&
    g.hasEdge(2, 4) && g.hasEdge(4, 3) && g.hasEdge(3, 1)
  )) return "DiGraphIntProp";
  for (int u : g.vertices()) {
    if (!(
      u >= 1 && u <= 4 &&
      g.degree(u) <= 1 &&
      g.vertexData(u) <= 10
    )) return "DiGraphIntVertex";
    for (int v : g.edges(u))
      if (!(
        v >= 1 && v <= 4 &&
        g.edgeData(u, v) == 10*u+v
      )) return "DiGraphIntEdge";
  }

  if (!(
    h.span()  == 4  &&
    h.order() == 4  &&
    h.size()  == 3  &&
    h.hasVertex('a')  && h.hasVertex('b')  && h.hasVertex('c')  && h.hasVertex('d') &&
    h.hasEdge('b', 'd') && h.hasEdge('d', 'c') && h.hasEdge('c', 'a')
  )) return "DiGraphCharProp";
  for (char u : h.vertices()) {
    if (!(
      u >= 'a' && u <= 'd' &&
      h.degree(u) <= 1 &&
      h.vertexData(u) <= 10
    )) return "DiGraphCharVertex";
    for (char v : h.edges(u))
      if (!(
        v >= 'a' && v <= 'd' &&
        h.edgeData(u, v) == 10*(u-'a'+1)+(v-'a'+1)
      )) return "DiGraphCharEdge";
  }
  return NULL;
}


const char* testCompactDiGraph() {
  CompactDiGraph<int, int, int> h;
  auto& g = h.base();
  h.addVertex(1, 10);
  h.addEdge(1, 2, 12);
  h.addEdge(2, 4, 24);
  h.addEdge(4, 3, 43);
  h.addEdge(3, 1, 31);
  h.removeEdge(1, 2);

  if (!(
    g.span()  == 4  &&
    g.order() == 4  &&
    g.size()  == 3  &&
    g.hasVertex(0)  && g.hasVertex(1)  && g.hasVertex(2)  && g.hasVertex(3) &&
    g.hasEdge(1, 3) && g.hasEdge(3, 2) && g.hasEdge(2, 0)
  )) return "CompactDiGraphBaseProp";
  for (int u : g.vertices()) {
    if (!(
      u >= 0 && u <= 3 &&
      g.degree(u) <= 1 &&
      g.vertexData(u) <= 10
    )) return "CompactDiGraphBaseVertex";
    for (int v : g.edges(u))
      if (!(
        v >= 0 && v <= 3 &&
        g.edgeData(u, v) == 10*(u+1)+(v+1)
      )) return "CompactDiGraphBaseEdge";
  }

  if (!(
    h.span()  == 4  &&
    h.order() == 4  &&
    h.size()  == 3  &&
    h.hasVertex(1)  && h.hasVertex(2)  && h.hasVertex(3)  && h.hasVertex(4) &&
    h.hasEdge(2, 4) && h.hasEdge(4, 3) && h.hasEdge(3, 1)
  )) return "CompactDiGraphIntProp";
  for (int u : h.vertices()) {
    if (!(
      u >= 1 && u <= 4 &&
      h.degree(u) <= 1 &&
      h.vertexData(u) <= 10
    )) return "CompactDiGraphIntVertex";
    for (int v : h.edges(u))
      if (!(
        v >= 1 && v <= 4 &&
        h.edgeData(u, v) == 10*u+v
      )) return "CompactDiGraphIntEdge";
  }
  return NULL;
}


const char* testCopy() {
  DiGraph<> g;
  CompactDiGraph<> h;
  g.addEdge(1, 2);
  g.addEdge(2, 4);
  g.addEdge(4, 3);
  g.addEdge(3, 1);
  copy(g, h);

  if (!(
    h.order() == 4  &&
    h.size()  == 4  &&
    h.hasEdge(1, 2) &&
    h.hasEdge(2, 4) &&
    h.hasEdge(4, 3) &&
    h.hasEdge(3, 1)
  //  g.vertexKeys()         == h.vertexKeys() &&
  //  g.vertexData()         == h.vertexData() &&
  //  g.edgeData()           == h.edgeData()   &&
  //  g.sourceOffsets()      == h.sourceOffsets() &&
  //  g.destinationIndices() == h.destinationIndices()
  )) return "copy";
  return NULL;
}


const char* testTranspose() {
  DiGraph<> g;
  DiGraph<> h;
  g.addEdge(1, 2);
  g.addEdge(2, 4);
  g.addEdge(4, 3);
  g.addEdge(3, 1);
  transpose(g, h);

  if (!(
   h.order() == 4  &&
   h.size()  == 4  &&
   h.hasEdge(2, 1) &&
   h.hasEdge(4, 2) &&
   h.hasEdge(3, 4) &&
   h.hasEdge(1, 3)
  )) return "transpose";
  return NULL;
}


void testAll() {
  vector<const char*> ts = {
    testCeilDiv(),
    testErase(),
    testInsert(),
    testFilter(),
    testFind(),
    testLowerBound(),
    testRange(),
    testTransform(),
    testFill(),
    testAdd(),
    testSum(),
    testDotProduct(),
    testErrorAbs(),
    testDiGraph(),
    testCompactDiGraph(),
    testCopy(),
    testTranspose()
  };
  int n = 0;
  for (auto& t : ts) {
    if (!t) continue;
    printf("ERROR: %s() failed!\n", t);
    n++;
  }
  printf("%d/%ld tests failed.\n", n, ts.size());
}
