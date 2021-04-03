#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include "DiGraph.h"

using std::ios;
using std::string;
using std::ifstream;
using std::istringstream;
using std::getline;




template <class G>
void readMtx(const char *pth, G& a) {
  ifstream f(pth);
  string b, ln;

  // read all data
  f.seekg(0, ios::end);
  b.resize(f.tellg());
  f.seekg(0);
  f.read((char*) b.data(), b.size());
  istringstream bs(b);

  // read rows, cols, size
  int r, c, sz;
  do { getline(bs, ln); }
  while (ln[0] == '%');
  istringstream ls(ln);
  ls >> r >> c >> sz;

  // read edges (from, to)
  while (getline(bs, ln)) {
    int u, v;
    ls = istringstream(ln);
    if (!(ls >> u >> v)) break;
    a.addEdge(u, v);
  }
}

auto readMtx(const char *pth) {
  DiGraph<> a; readMtx(pth, a);
  return a;
}
