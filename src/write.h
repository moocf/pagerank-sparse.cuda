#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <ostream>
#include <sstream>
#include <fstream>

using std::string;
using std::vector;
using std::unordered_map;
using std::ostream;
using std::stringstream;
using std::ofstream;




// WRITE
// -----

template <class T>
void write(ostream& a, T *x, int N) {
  a << "{";
  for (int i=0; i<N; i++)
    a << " " << x[i];
  a << " }";
}

template <class T>
void write(ostream& a, vector<T>& x) {
  write(a, x.data(), x.size());
}

template <class K, class T>
void write(ostream& a, unordered_map<K, T>& x) {
  a << "{\n";
  for (auto& p : x)
    a << "  " << p.first << " => " << p.second << "\n";
  a << "}";
}




template <class T>
void write(ostream& a, T *x, int R, int C) {
  a << "{\n";
  for (int r=0; r<R; r++) {
    for (int c=0; c<C; c++)
      a << "  " << GET2D(x, r, c, C);
    a << "\n";
  }
  a << "}";
}

template <class T>
void write(ostream& a, vector<vector<T>>& x) {
  a << "{\n";
  for (auto& v : x) {
    a << "  "; write(a, v);
  }
  a << "}";
}




template <class G>
void write(ostream& a, G& x, bool all=false) {
  a << "order: " << x.order() << " size: " << x.size();
  if (!all) { a << " {}"; return; }
  a << " {\n";
  for (auto u : x.vertices()) {
    a << "  " << u << " ->";
    for (auto v : x.edges(u))
      a << " " << v;
    a << "\n";
  }
  a << "}";
}




// WRITE-MTX
// ---------

template <class G>
void writeMtx(ostream& a, G& x) {
  a << "%%MatrixMarket matrix coordinate integer asymmetric\n";
  a << x.order() << " " << x.order() << " " << x.size() << "\n";
  for (auto u : x.vertices()) {
    for (auto v : x.edges(u))
      a << u << " " << v << " " << x.edgeData(u) << "\n";
  }
}

template <class G>
void writeMtx(string pth, G& x) {
  string s0; stringstream s(s0);
  writeMtx(s, x);
  ofstream f(pth);
  f << s.rdbuf();
  f.close();
}
