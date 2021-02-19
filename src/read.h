#pragma once
#include <fstream>
#include <sstream>
#include <string>

using std::string;
using std::ifstream;
using std::istringstream;
using std::getline;




template <class G>
void readMtx(string pth, G& a) {
  int r, c, sz;
  ifstream f(pth);
  string ln;

  getline(f, ln);
  getline(f, ln);
  istringstream ls(ln);
  ls >> r >> c >> sz;
  while (getline(f, ln)) {
    int i, j; float w;
    ls = istringstream(ln);
    if (!(ls >> i >> j >> w)) break;
    if (w > 0) a.addEdge(i, j);
  }
}
