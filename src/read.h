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

  do { getline(f, ln); }
  while (ln[0] == '%');
  istringstream ls(ln);
  ls >> r >> c >> sz;
  while (getline(f, ln)) {
    int u, v;
    ls = istringstream(ln);
    if (!(ls >> u >> v)) break;
    a.addEdge(u, v);
  }
}
