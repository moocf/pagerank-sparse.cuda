#pragma once
#include <string>
#include <istream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "DiGraph.h"

using std::ios;
using std::string;
using std::istream;
using std::ifstream;
using std::istringstream;
using std::getline;
using std::max;




// READ-FILE
// ---------

string readFile(const char *pth) {
  string a; ifstream f(pth);
  f.seekg(0, ios::end);
  a.resize(f.tellg());
  f.seekg(0);
  f.read((char*) a.data(), a.size());
  return a;
}




// READ-MTX
// --------

template <class G>
void readMtx(G& a, istream& s) {
  string ln, h0, h1, h2, h3, h4;

  // read header
  while (1) {
    getline(s, ln);
    if (ln.find('%')!=0) break;
    if (ln.find("%%")!=0) continue;
    istringstream ls(ln);
    ls >> h0 >> h1 >> h2 >> h3 >> h4;
  }
  cout << "h1: " << h1 << " h2: " << h2 << "\n";
  if (h1!="matrix" || h2!="coordinate") return;
  bool sym = h4=="symmetric" || h4=="skew-symmetric";

  // read rows, cols, size
  int r, c, sz;
  istringstream ls(ln);
  ls >> r >> c >> sz;
  int n = max(r, c);
  for (int u=1; u<=n; u++)
    a.addVertex(u);

  // read edges (from, to)
  while (getline(s, ln)) {
    int u, v;
    ls = istringstream(ln);
    if (!(ls >> u >> v)) break;
    a.addEdge(u, v);
    if (sym) a.addEdge(v, u);
  }
}

auto readMtx(istream& s) {
  DiGraph<> a; readMtx(a, s);
  return a;
}


template <class G>
void readMtx(G& a, const char *pth) {
  string buf = readFile(pth);
  istringstream s(buf);
  return readMtx(a, s);
}

auto readMtx(const char *pth) {
  DiGraph<> a; readMtx(a, pth);
  return a;
}
