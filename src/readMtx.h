#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include "DiGraphTemp.h"

using std::string;
using std::ifstream;
using std::istringstream;
using std::getline;




template <class K, class V, class E>
void readMtx(DiGraphTemp<K, V, E>& a, string pth) {
  string ln;
  ifstream f(pth);

  // skip 1st line
  getline(f, ln);

  // read 2nd line
  int r, c, sz;
  getline(f, ln);
  istringstream ls(ln);
  ls >> r >> c >> sz;

  // read remaining lines (edges)
  while (getline(f, ln)) {
    int i, j; float w;
    ls = istringstream(ln);
    if (!(ls >> i >> j >> w)) break;
    if (w > 0) a.addEdge(i, j);
  }
}
