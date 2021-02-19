#pragma once
#include <fstream>
#include <string>

using std::string;
using std::ofstream;




template <class G>
void writeMtx(string pth, const G& g) {
  ofstream f(pth);
  f << "%%MatrixMarket matrix coordinate integer asymmetric\n";
  f << g.order() << " " << g.order() << " " << g.size() << "\n";
  for (auto&& u : g.vertices()) {
    for (auto&& v : g.edges(u))
      f << u << " " << v << " " << (int) g.edgeData(u) << "\n";
  }
  f.close();
}
