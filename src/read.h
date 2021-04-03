#pragma once
#include <fstream>
#include <sstream>
#include <string>

using std::ios;
using std::string;
using std::ifstream;
using std::istringstream;
using std::getline;




template <class G>
void readMtx(string pth, G& a) {
  int r, c, sz;
  string b;
  ifstream f(pth);

  f.seekg(0, ios::end);
  b.resize(f.tellg());

  f.seekg(0);
  f.read((char*) b.data(), b.size());

  string ln;
  istringstream bs(b);
  do { getline(bs, ln); }
  while (ln[0] == '%');

  istringstream ls(ln);
  ls >> r >> c >> sz;

  while (getline(bs, ln)) {
    int u, v;
    ls = istringstream(ln);
    if (!(ls >> u >> v)) break;
    a.addEdge(u, v);
  }
}
